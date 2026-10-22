"""Р4. Пространство действий агента: каждая гипотеза — это новый Dataset и/или ModelConfig.

Действие возвращает Candidate (что попробовать) или None (нечего пробовать).
Обучение и оценку делает research_loop: обучение — ML-ядро (Р3), решение — validation (Р5).
Добавить действие = функция + запись в ACTIONS.

Что улучшать (Р4): анализ ошибок модели (где OOF-ошибка больше всего) в промпте,
действия для предобработки (выбросы, пропуски), отбор признаков по permutation importance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import config
from agent.llm import extract_code, llm
from agent.memory import Memory
from agent.sandbox import run_feature_code
from core.contracts import Dataset, ExperimentResult, Folds, ModelConfig, TaskSpec
from ml.tuner import tune_lgbm
from runtime.clock import Clock
from runtime.logger import log
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


# ---------- признаки ----------

def _full_frame(ds: Dataset) -> pd.DataFrame:
    full = pd.concat([ds.X, ds.X_test], ignore_index=True)
    for col in ds.cat_cols:
        full[col] = full[col].astype(object)
    return full


def _with_new_columns(ds: Dataset, new: pd.DataFrame) -> Dataset | None:
    new = new[[c for c in new.columns if c not in ds.X.columns]].reset_index(drop=True)
    cat_cols = list(ds.cat_cols)
    for col in list(new.columns):
        if pd.api.types.is_numeric_dtype(new[col]) or pd.api.types.is_bool_dtype(new[col]):
            new[col] = pd.to_numeric(new[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            new[col] = new[col].astype("string").astype("category")
            cat_cols.append(col)
        if new[col].nunique(dropna=False) <= 1:
            new = new.drop(columns=[col])
            cat_cols = [c for c in cat_cols if c != col]
    if new.empty:
        return None
    n = len(ds.X)
    return Dataset(
        X=pd.concat([ds.X, new.iloc[:n].reset_index(drop=True)], axis=1),
        y=ds.y,
        X_test=pd.concat([ds.X_test, new.iloc[n:].reset_index(drop=True)], axis=1),
        test_ids=ds.test_ids,
        cat_cols=cat_cols,
    )


def _top_numeric(state: ResearchState, k: int) -> list[str]:
    return [c for c in state.best.feature_importance if c in state.ds.X.columns
            and c not in state.ds.cat_cols][:k]


def _recipe_freq(state: ResearchState, full: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"{c}__freq": full[c].map(full[c].value_counts(normalize=True)).astype(float)
                         for c in state.ds.cat_cols})


def _recipe_group_diff(state: ResearchState, full: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in state.best.feature_importance if c in state.ds.cat_cols][:2]
    out = {}
    for cat in cats:
        for num in _top_numeric(state, 3):
            mean = full.groupby(cat)[num].transform("mean")
            out[f"{num}__minus_mean_by_{cat}"] = full[num] - mean
            out[f"{num}__ratio_mean_by_{cat}"] = full[num] / (mean.abs() + 1e-6)
    return pd.DataFrame(out)


def _recipe_ratios(state: ResearchState, full: pd.DataFrame) -> pd.DataFrame:
    top = _top_numeric(state, 4)
    return pd.DataFrame({f"{a}__div__{b}": full[a] / (full[b].abs() + 1e-6)
                         for i, a in enumerate(top) for b in top[i + 1:]})


RECIPES: dict[str, tuple[str, Callable]] = {
    "freq": ("Частотное кодирование категорий", _recipe_freq),
    "group_diff": ("Отклонение важных чисел от среднего по важным категориям", _recipe_group_diff),
    "ratios": ("Отношения пар самых важных числовых признаков", _recipe_ratios),
}

FEATURES_SYSTEM = """Ты опытный ML-инженер по табличным данным. Предложи ОДНУ гипотезу новых признаков и напиши код.
Правила:
- ровно одна функция `def transform_features(df: pd.DataFrame) -> pd.DataFrame`; доступны только pd и np;
- df — признаки train и test вместе, таргета в нём нет; верни df с ДОБАВЛЕННЫМИ колонками, исходные не удаляй;
- 2-8 новых числовых признаков; деление защищай: a / (b.abs() + 1e-6); никаких print, файлов и сети;
- опирайся на смысл колонок и предметную область, не повторяй уже проверенные идеи.
Формат ответа: 1-2 предложения гипотезы, затем код в блоке ```python```."""


def _features_llm(state: ResearchState) -> Candidate | None:
    top = ", ".join(list(state.best.feature_importance)[:12])
    user = (f"Задача:\n{state.spec.text[:2500]}\n\nЗаметки: {state.spec.notes}\n\n"
            f"Профиль данных:\n{state.profile[:5000]}\n\n"
            f"Метрика {state.spec.metric}, текущий лучший скор {state.best.score:.5f}.\n"
            f"Самые важные признаки: {top}\n"
            f"Признаки со сдвигом между train и test (осторожно с ними): {state.shifted}\n\n"
            f"История экспериментов:\n{state.memory.summary()}")
    answer = llm.ask(FEATURES_SYSTEM, user, max_tokens=2000)
    code = extract_code(answer)
    if not code:
        return None
    hypothesis = answer.split("```")[0].strip().replace("\n", " ")[:300] or "Признаки от LLM"
    out, error = run_feature_code(code, _full_frame(state.ds))
    if error:
        raise RuntimeError(f"Код признаков упал: {error}")
    ds = _with_new_columns(state.ds, out)
    return Candidate(ds, state.cfg.copy(), hypothesis, code) if ds else None


def _features_heuristic(state: ResearchState) -> Candidate | None:
    for key, (description, recipe) in RECIPES.items():
        if key in state.used_recipes:
            continue
        state.used_recipes.add(key)
        new = recipe(state, _full_frame(state.ds))
        ds = _with_new_columns(state.ds, new) if not new.empty else None
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
