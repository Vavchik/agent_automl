"""Р1. Порядок стадий агента. Каждая стадия — вызов в свою папку (свою роль).

1. Понимание задачи (task/, Р2)          4. Бейзлайн и сабмит v0 (ml/, Р3) = корень дерева
2. Признаки и типы (task/, Р2)           5. AutoResearch: поиск по дереву решений (agent/, Р4)
3. Схема валидации (validation/, Р5)     6. Финал: бленд моделей, AutoGluon, ансамбль (ml/, Р3)
После шага 4 submission.csv существует всегда; дальше он только улучшается.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from agent.actions import ResearchState
from agent.llm import llm
from agent.memory import Memory, Record
from agent.research_loop import run_research
from agent.tree import Node, SolutionTree
from core.contracts import Dataset, ExperimentResult, ModelConfig, TaskSpec
from ml import autogluon_stage
from ml.ensemble import greedy_blend
from ml.postprocess import final_prediction
from ml.trainer import target_array, train_cv
from runtime.clock import Clock
from runtime.logger import log, setup_logger
from task.dtypes import prepare_dataset
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
    log.info(llm.describe())

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

    # 4. Бейзлайн (Р3) -> сабмит v0, корень дерева
    cfg = ModelConfig(models=list(config.RESEARCH_MODELS))
    baseline = train_cv(ds, spec, folds, cfg, name="baseline")
    submit(spec, ds, baseline)
    root = Node(id=0, parent_id=None, action="baseline", hypothesis=f"{'+'.join(cfg.models)}, {scheme}",
                result=baseline, ds=ds, cfg=cfg, lineage=frozenset())
    tree = SolutionTree(root, spec.metric)

    # 5. AutoResearch по дереву (Р4)
    memory = Memory(spec.task_dir / "experiments.jsonl")
    memory.log(Record(0, "baseline", root.hypothesis, baseline.score, None, True, "", baseline.seconds, node_id=0))
    state = ResearchState(spec=spec, folds=folds, memory=memory, clock=clock, tree=tree, shifted=shifted,
                          results=[baseline])
    use_ag = config.AUTOGLUON_TIME_SEC > 0 and autogluon_stage.available()
    reserve = (config.AUTOGLUON_TIME_SEC if use_ag else 0.0)
    try:
        run_research(state, on_new_best=lambda node: submit(spec, node.ds, node.result),
                     reserve_sec=reserve + config.FINAL_TIME_FACTOR * baseline.seconds)
    except Exception as e:
        log.warning(f"AutoResearch прерван: {e}")

    best = tree.best
    best_result = best.result

    # 6. Финал (Р3)
    def try_accept(result: ExperimentResult | None, label: str) -> None:
        nonlocal best_result
        if result is None:
            return
        state.results.append(result)
        accepted, reason = should_accept(result, best_result, spec.metric)
        log.info(f"{label}: {result.score:.5f} — {'принят' if accepted else 'отклонён'} ({reason})")
        if accepted:
            best_result = result
            submit(spec, best.ds, result)

    try:
        final_cfg = best.cfg.copy()
        final_cfg.models = list(config.FINAL_MODELS)
        if final_cfg.models != best.cfg.models and clock.has(config.FINAL_TIME_FACTOR * best.result.seconds + reserve):
            try_accept(train_cv(best.ds, spec, folds, final_cfg, name="final_blend"), "Финальный бленд моделей")
    except Exception as e:
        log.warning(f"Финальный бленд не удался: {e}")

    if use_ag and clock.remaining() > 10 * 60:
        ag_time = min(config.AUTOGLUON_TIME_SEC, clock.remaining() - 5 * 60)
        try_accept(autogluon_stage.run_autogluon(best.ds, spec, folds, ag_time), "AutoGluon на наших фолдах")

    try:
        y, _ = target_array(ds, spec)
        try_accept(greedy_blend(state.results, spec, y, folds), "Ансамбль экспериментов")
    except Exception as e:
        log.warning(f"Ансамбль не удался: {e}")

    best_path = " -> ".join(f"#{n.id}:{n.action}" for n in tree.path(best))
    report = {
        "task_type": spec.task_type, "metric": spec.metric, "output": spec.output,
        "validation": scheme, "adversarial_auc": adv_auc,
        "baseline_score": baseline.score, "best_score": best_result.score, "best": best_result.name,
        "best_path": best_path, "nodes": len(tree.nodes),
        "improving_nodes": sum(n.improving for n in tree.nodes.values()),
        "failed_nodes": sum(bool(n.error) for n in tree.nodes.values()),
        "llm": llm.describe(), "llm_calls": llm.calls, "llm_tokens": llm.used_tokens,
        "seconds": round(clock.elapsed()),
    }
    (spec.task_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Готово: {json.dumps(report, ensure_ascii=False)}")
    return spec.task_dir / "submission.csv"
