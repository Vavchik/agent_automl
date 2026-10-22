"""Р1. Порядок стадий агента. Каждая стадия — вызов в свою папку (свою роль).

1. Понимание задачи (task/, Р2)          4. Бейзлайн и сабмит v0 (ml/, Р3)
2. Признаки и типы (task/, Р2)           5. AutoResearch (agent/, Р4)
3. Схема валидации (validation/, Р5)     6. Ансамбль и финальный сабмит (ml/, Р3)
После шага 4 submission.csv существует всегда; дальше он только улучшается.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config
from agent.actions import ResearchState
from agent.llm import llm
from agent.memory import Memory, Record
from agent.research_loop import run_research
from core.contracts import Dataset, ExperimentResult, ModelConfig, TaskSpec
from ml.ensemble import greedy_blend
from ml.postprocess import final_prediction
from ml.trainer import target_array, train_cv
from runtime.clock import Clock
from runtime.logger import log, setup_logger
from task.dtypes import prepare_dataset, profile_text
from task.llm_parser import refine_with_llm
from task.loader import load_task
from task.submission import write_submission
from validation.acceptance import should_accept
from validation.adversarial import adversarial_check
from validation.splits import make_folds


def submit(spec: TaskSpec, ds: Dataset, result: ExperimentResult) -> Path:
    y, _ = target_array(ds, spec)
    pred, classes = final_prediction(spec, result, y)
    path = write_submission(spec, ds.test_ids, pred, classes)
    log.info(f"submission.csv обновлён: {result.name}, {spec.metric} = {result.score:.5f}")
    return path


def run(task_md: Path) -> Path:
    clock = Clock()
    task_md = Path(task_md).resolve()
    setup_logger(task_md.parent)
    old = task_md.parent / "submission.csv"
    if old.exists():
        old.unlink()                      # чтобы не принять старый файл за результат этого запуска

    # 1-2. Задача и признаки (Р2)
    spec, train, test = load_task(task_md)
    spec = refine_with_llm(spec, train)
    if config.FAST and len(train) > config.FAST_MAX_TRAIN_ROWS:
        frac = config.FAST_MAX_TRAIN_ROWS / len(train)
        train = (train.groupby(spec.target_col, group_keys=False).sample(frac=frac, random_state=config.SEED)
                 if spec.is_classification else train.sample(frac=frac, random_state=config.SEED))
        log.info(f"Быстрый режим: train уменьшен до {len(train)} строк")
    ds = prepare_dataset(train, test, spec)

    # 3. Валидация (Р5)
    folds, scheme = make_folds(ds, spec)
    try:
        adv_auc, shifted = adversarial_check(ds, exclude=[spec.time_col] if spec.time_col else None)
    except Exception as e:
        log.warning(f"Adversarial-проверка не удалась: {e}")
        adv_auc, shifted = None, []

    # 4. Бейзлайн (Р3) -> сабмит v0
    cfg = ModelConfig(models=list(config.RESEARCH_MODELS))
    baseline = train_cv(ds, spec, folds, cfg, name="baseline")
    submit(spec, ds, baseline)

    # 5. AutoResearch (Р4)
    memory = Memory(spec.task_dir / "experiments.jsonl")
    memory.log(Record(0, "baseline", f"{'+'.join(cfg.models)}, {scheme}", baseline.score, None, True, "",
                      baseline.seconds))
    state = ResearchState(spec=spec, ds=ds, cfg=cfg, best=baseline, folds=folds, memory=memory, clock=clock,
                          profile=profile_text(ds, spec), shifted=shifted, results=[baseline])
    try:
        run_research(state, on_new_best=lambda r, s: submit(spec, s.ds, r))
    except Exception as e:
        log.warning(f"AutoResearch прерван: {e}")

    # 6. Финал (Р3): бленд нескольких моделей на лучших признаках + ансамбль экспериментов
    try:
        final_cfg = state.cfg.copy()
        final_cfg.models = list(config.FINAL_MODELS)
        if final_cfg.models != state.cfg.models and clock.has(config.FINAL_TIME_FACTOR * state.best.seconds):
            final = train_cv(state.ds, spec, folds, final_cfg, name="final_blend")
            state.results.append(final)
            accepted, reason = should_accept(final, state.best, spec.metric)
            log.info(f"Финальный бленд {'принят' if accepted else 'отклонён'} ({reason})")
            if accepted:
                state.best, state.cfg = final, final_cfg
                submit(spec, state.ds, final)
    except Exception as e:
        log.warning(f"Финальный бленд не удался: {e}")
    try:
        y, _ = target_array(ds, spec)
        blend = greedy_blend(state.results, spec, y, folds)
        if blend is not None:
            accepted, reason = should_accept(blend, state.best, spec.metric)
            log.info(f"Ансамбль {blend.name}: {blend.score:.5f} — {'принят' if accepted else 'отклонён'} ({reason})")
            if accepted:
                state.best = blend
                submit(spec, state.ds, blend)
    except Exception as e:
        log.warning(f"Ансамбль не удался: {e}")

    report = {
        "task_type": spec.task_type, "metric": spec.metric, "output": spec.output,
        "validation": scheme, "adversarial_auc": adv_auc,
        "baseline_score": baseline.score, "best_score": state.best.score, "best": state.best.name,
        "experiments": len(memory.records), "accepted": sum(r.accepted for r in memory.records) - 1,
        "llm_tokens": llm.used_tokens, "seconds": round(clock.elapsed()),
    }
    (spec.task_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Готово: {json.dumps(report, ensure_ascii=False)}")
    return spec.task_dir / "submission.csv"
