"""Р4. Пространство действий агента: каждая гипотеза — это новый Dataset и/или ModelConfig.

Действие возвращает Candidate (что попробовать) или None (нечего пробовать).
Обучение и оценку делает research_loop: обучение — ML-ядро (Р3), решение — validation (Р5).
Добавить действие = функция + запись в ACTIONS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from agent.llm import extract_code, extract_hypothesis, extract_thinking, llm
from agent.memory import Memory
from agent.sandbox import run_feature_code
from core.contracts import Dataset, ExperimentResult, Folds, ModelConfig, TaskSpec
from ml.preprocess import compute_oof_target_encoding
from ml.tuner import tune_lgbm
from runtime.clock import Clock
from runtime.logger import log
from task.dtypes import build_schema_json
from task.metrics import METRICS


@dataclass
class Candidate:
    ds: Dataset
    cfg: ModelConfig
    hypothesis: str
    code: str = ""


@dataclass
class ResearchState:
    spec: TaskSpec
    ds: Dataset
    cfg: ModelConfig
    best: ExperimentResult
    folds: Folds
    memory: Memory
    clock: Clock
    profile: str = ""
    shifted: list[str] = field(default_factory=list)       # признаки со сдвигом train/test
    results: list[ExperimentResult] = field(default_factory=list)
    used_recipes: set[str] = field(default_factory=set)
    tree: Any = None


# ---------- признаки ----------

def _with_new_columns_separate(ds: Dataset, new_tr: pd.DataFrame, new_te: pd.DataFrame) -> Dataset | None:
    new_cols = [c for c in new_tr.columns if c not in ds.X.columns and c in new_te.columns]
    if not new_cols:
        return None
    new_tr = new_tr[new_cols].copy().reset_index(drop=True)
    new_te = new_te[new_cols].copy().reset_index(drop=True)

    cat_cols = list(ds.cat_cols)
    valid_cols = []
    for col in new_cols:
        tr_s = new_tr[col]
        te_s = new_te[col]
        if pd.api.types.is_numeric_dtype(tr_s) or pd.api.types.is_bool_dtype(tr_s):
            new_tr[col] = pd.to_numeric(tr_s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            new_te[col] = pd.to_numeric(te_s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            combined = pd.concat([tr_s.astype("string"), te_s.astype("string")], ignore_index=True)
            cat_type = pd.CategoricalDtype(categories=combined.dropna().unique())
            new_tr[col] = tr_s.astype(cat_type)
            new_te[col] = te_s.astype(cat_type)
            cat_cols.append(col)

        if new_tr[col].nunique(dropna=False) > 1:
            valid_cols.append(col)
        else:
            cat_cols = [c for c in cat_cols if c != col]

    if not valid_cols:
        return None

    return Dataset(
        X=pd.concat([ds.X.reset_index(drop=True), new_tr[valid_cols]], axis=1),
        y=ds.y.reset_index(drop=True),
        X_test=pd.concat([ds.X_test.reset_index(drop=True), new_te[valid_cols]], axis=1),
        test_ids=ds.test_ids,
        cat_cols=cat_cols,
    )


def _with_new_columns(ds: Dataset, new: pd.DataFrame) -> Dataset | None:
    n = len(ds.X)
    return _with_new_columns_separate(ds, new.iloc[:n].reset_index(drop=True), new.iloc[n:].reset_index(drop=True))


def _top_numeric(state: ResearchState, k: int) -> list[str]:
    return [c for c in state.best.feature_importance if c in state.ds.X.columns
            and c not in state.ds.cat_cols][:k]


def _recipe_freq(state: ResearchState) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_out = {}
    te_out = {}
    for c in state.ds.cat_cols:
        freq = state.ds.X[c].value_counts(normalize=True)
        tr_out[f"{c}__freq"] = state.ds.X[c].map(freq).astype(float).fillna(0.0)
        te_out[f"{c}__freq"] = state.ds.X_test[c].map(freq).astype(float).fillna(0.0)
    return pd.DataFrame(tr_out, index=state.ds.X.index), pd.DataFrame(te_out, index=state.ds.X_test.index)


def _recipe_group_diff(state: ResearchState) -> tuple[pd.DataFrame, pd.DataFrame]:
    cats = [c for c in state.best.feature_importance if c in state.ds.cat_cols][:2]
    tr_out = {}
    te_out = {}
    for cat in cats:
        for num in _top_numeric(state, 3):
            mean = state.ds.X.groupby(cat)[num].mean()
            tr_mean = state.ds.X[cat].map(mean)
            te_mean = state.ds.X_test[cat].map(mean)
            tr_out[f"{num}__diff_mean_{cat}"] = state.ds.X[num] - tr_mean
            tr_out[f"{num}__ratio_mean_{cat}"] = state.ds.X[num] / (tr_mean.abs() + 1e-6)
            te_out[f"{num}__diff_mean_{cat}"] = state.ds.X_test[num] - te_mean
            te_out[f"{num}__ratio_mean_{cat}"] = state.ds.X_test[num] / (te_mean.abs() + 1e-6)
    return pd.DataFrame(tr_out, index=state.ds.X.index), pd.DataFrame(te_out, index=state.ds.X_test.index)


def _recipe_ratios(state: ResearchState) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = _top_numeric(state, 4)
    tr_out = {}
    te_out = {}
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            tr_out[f"{a}__div__{b}"] = state.ds.X[a] / (state.ds.X[b].abs() + 1e-6)
            te_out[f"{a}__div__{b}"] = state.ds.X_test[a] / (state.ds.X_test[b].abs() + 1e-6)
    return pd.DataFrame(tr_out, index=state.ds.X.index), pd.DataFrame(te_out, index=state.ds.X_test.index)


def _recipe_target_enc(state: ResearchState) -> tuple[pd.DataFrame, pd.DataFrame]:
    return compute_oof_target_encoding(
        state.ds.X, state.ds.y, state.ds.X_test, state.ds.cat_cols, state.folds, state.spec.task_type
    )


RECIPES: dict[str, tuple[str, Callable]] = {
    "freq": ("Частотное кодирование категорий", _recipe_freq),
    "target_enc": ("OOF Target Encoding категориальных признаков со сглаживанием", _recipe_target_enc),
    "group_diff": ("Отклонение важных чисел от среднего по важным категориям", _recipe_group_diff),
    "ratios": ("Отношения пар самых важных числовых признаков", _recipe_ratios),
}

FEATURES_SYSTEM = """Ты опытный Senior ML/Kaggle Grandmaster инженер по табличным данным.
Твоя цель: предложить сильную гипотезу генерации признаков для улучшения метрики и написать воспроизводимый Python-код.

ВАЖНО: Рассуждай предельно кратко (не более 2-3 предложений). Сразу выводи теги <hypothesis> и <python_code>.

Правила:
1. Сигнатура функции СТРОГО:
   def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
2. Защита от утечек (No Data Leakage):
   - Все статистики (mean, median, quantiles, std, группы) вычисляются ТОЛЬКО на train_df и проецируются (через map/merge/reindex) на test_df.
   - Таргета в train_df и test_df нет.
3. Сохранение данных:
   - Не удаляй существующие колонки, верни train_df и test_df с ДОБАВЛЕННЫМИ 2-8 новыми числовыми признаками.
4. Защита математики:
   - Деления защищай: a / (b.abs() + 1e-6) для pd.Series или a / (abs(b) + 1e-6) для скалярных float/int
   - Логарифмы: np.log1p(np.maximum(0, x))
5. Совместимость с Pandas 3.0:
   - Не используй inplace=True.
   - Не присваивай новые категории через fillna(0) в категориальные столбцы.
6. Изоляция:
   - Доступны ТОЛЬКО pandas (pd) и numpy (np). Никаких print, os, sys, файлов и сети.
   - Внутри тегов <python_code> пиши ТОЛЬКО валидный Python-код (никакого русского текста рассуждений).

Формат ответа СТРОГО в XML-тегах:
<hypothesis>
1-2 предложения с формулировкой гипотезы.
</hypothesis>
<python_code>
def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # расчет признаков
    return train_df, test_df
</python_code>"""


def _features_llm(state: ResearchState) -> Candidate | None:
    schema_json = build_schema_json(state.ds, state.spec, state.best.feature_importance, state.shifted)
    user_prompt = (
        f"Задача и профиль данных (Schema-First JSON):\n{schema_json}\n\n"
        f"Метрика: {state.spec.metric}, текущий лучший скор на CV: {state.best.score:.5f}.\n\n"
        f"История экспериментов:\n{state.memory.summary(last=6)}\n\n"
        f"Предложи сильную, небанальную гипотезу признаков с учетом природы задачи и выведи XML-теги."
    )
    answer = llm.ask(FEATURES_SYSTEM, user_prompt, max_tokens=3500)
    code = extract_code(answer)
    if not code:
        log.warning("LLM не вернула валидный код функции transform_features")
        return None
    hypothesis = extract_hypothesis(answer)
    thinking = extract_thinking(answer)
    if thinking:
        log.info(f"LLM Thinking: {thinking[:150]}...")

    res, error = run_feature_code(code, state.ds.X, state.ds.X_test)

    # Self-debugging loop (до 2 попыток исправления ошибок выполнения)
    if error and llm.available():
        for attempt in range(2):
            log.warning(f"Ошибка выполнения кода признаков от LLM: {error[:200]}. Self-debugging ({attempt + 1}/2)...")
            fix_system = (
                "Ты Senior Python/Pandas инженер. Функция transform_features вызвала ошибку при выполнении.\n"
                "Твоя задача — точечно исправить ошибку в коде, сохранив математическую идею гипотезы.\n"
                "Выведи исправленный код строго в тегах <python_code>...</python_code> без текста рассуждений."
            )
            fix_user = (
                f"Исходный код:\n```python\n{code}\n```\n\n"
                f"Трассировка ошибки (stderr):\n```\n{error}\n```\n\n"
                f"Правила исправления:\n"
                f"1. Сигнатура: def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]\n"
                f"2. Все агрегаты вычисляются на train_df и проецируются на test_df\n"
                f"3. Деление строго через a / (b.abs() + 1e-6) для Series или a / (abs(b) + 1e-6) для скаляров\n"
                f"4. Не модифицируй категориальные столбцы через fillna со значениями не из категорий\n"
                f"5. Возвращай копии (train_df, test_df) с новыми признаками\n"
                f"6. Пиши ТОЛЬКО чистый исполняемый код функции без рассуждений на русском языке"
            )
            fix_answer = llm.ask(fix_system, fix_user, max_tokens=3500)
            fixed_code = extract_code(fix_answer)
            if not fixed_code:
                break
            code = fixed_code
            res, error = run_feature_code(code, state.ds.X, state.ds.X_test)
            if not error:
                log.info(f"Self-debugging успешно исправил код на попытке {attempt + 1}!")
                break

    if error or res is None:
        log.warning(f"Код признаков от LLM не удалось исполнить: {error}")
        return None

    out_train, out_test = res
    ds = _with_new_columns_separate(state.ds, out_train, out_test)
    if ds is None:
        log.warning("Функция transform_features не добавила новых валидных признаков")
        return None
    return Candidate(ds, state.cfg.copy(), hypothesis, code)


def _features_heuristic(state: ResearchState) -> Candidate | None:
    for key, (description, recipe) in RECIPES.items():
        if key in state.used_recipes:
            continue
        state.used_recipes.add(key)
        new_tr, new_te = recipe(state)
        ds = _with_new_columns_separate(state.ds, new_tr, new_te) if not new_tr.empty else None
        if ds is not None:
            return Candidate(ds, state.cfg.copy(), f"[эвристика] {description}")
    return None


def features(state: ResearchState) -> Candidate | None:
    if llm.available():
        candidate = _features_llm(state)
        if candidate is not None:
            return candidate
    return _features_heuristic(state)


# ---------- модель и таргет ----------

def target_log(state: ResearchState) -> Candidate:
    cfg = state.cfg.copy()
    cfg.target_transform = "log1p"
    return Candidate(state.ds, cfg, "Обучать на log1p(таргета): длинный правый хвост")


def class_balance(state: ResearchState) -> Candidate:
    cfg = state.cfg.copy()
    cfg.params.setdefault("lgbm", {})["class_weight"] = "balanced"
    cfg.params.setdefault("catboost", {})["auto_class_weights"] = "Balanced"
    return Candidate(state.ds, cfg, "Балансировка весов классов в моделях: метрика по редким классам")


def tune(state: ResearchState) -> Candidate | None:
    budget = min(config.TUNER_TIME_SEC, 0.3 * state.clock.remaining())
    params = tune_lgbm(state.ds, state.spec, state.folds, state.cfg, budget)
    if not params:
        return None
    cfg = state.cfg.copy()
    cfg.params["lgbm"] = params
    return Candidate(state.ds, cfg, f"Подобранные Optuna параметры LightGBM: {params}")


def drop_weak(state: ResearchState) -> Candidate | None:
    importance = pd.Series(state.best.feature_importance)
    share = importance / max(importance.sum(), 1e-12)
    weak = list(share[share < 0.002].index)[: int(0.3 * len(share))]
    if not weak:
        return None
    ds = Dataset(state.ds.X.drop(columns=weak), state.ds.y, state.ds.X_test.drop(columns=weak),
                 state.ds.test_ids, [c for c in state.ds.cat_cols if c not in weak])
    return Candidate(ds, state.cfg.copy(), f"Убрать {len(weak)} слабых признаков: {weak[:10]}")


@dataclass
class Action:
    name: str
    description: str
    run: Callable[[ResearchState], Candidate | None]
    can_run: Callable[[ResearchState], bool]
    repeatable: bool = False


def _label_metric(state: ResearchState) -> bool:
    return state.spec.is_classification and METRICS[state.spec.metric].needs == "label"


ACTIONS: dict[str, Action] = {a.name: a for a in [
    Action("features", "новые признаки (LLM пишет код, без LLM — готовые рецепты)", features,
           lambda s: llm.available() or len(s.used_recipes) < len(RECIPES), repeatable=True),
    Action("target_log", "обучение на log1p(таргета) для неотрицательного таргета с хвостом", target_log,
           lambda s: s.spec.task_type == "regression" and s.cfg.target_transform is None
           and s.ds.y.min() >= 0 and s.ds.y.skew() > 0.5),
    Action("class_balance", "веса классов в моделях (для Macro F1 и других метрик по меткам)", class_balance,
           _label_metric),
    Action("tune_lgbm", "подбор гиперпараметров LightGBM (Optuna, дорого по времени)", tune,
           lambda s: "lgbm" in s.cfg.models and s.clock.has(2 * config.TUNER_TIME_SEC)),
    Action("drop_weak", "удалить признаки с почти нулевой важностью", drop_weak,
           lambda s: s.ds.X.shape[1] > 10),
]}
