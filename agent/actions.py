"""Р4. Пространство действий агента. Действие применяется к узлу дерева (parent) и возвращает
Candidate — новые признаки и/или настройки обучения — или None, если пробовать нечего.

Обучение делает ML-ядро (Р3), решение о принятии — validation (Р5), место в дереве — agent/tree.py.
Добавить действие = функция run(state, parent) + запись в ACTIONS.

Что улучшать (Р4): действия для выбросов и пропусков, отбор признаков по permutation importance,
OOF target encoding как отдельная гипотеза.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import config
from agent.analysis import error_analysis
from agent.llm import extract_code, extract_tag, llm
from agent.memory import Memory
from agent.sandbox import run_feature_code
from agent.tree import Node, SolutionTree
from core.contracts import Dataset, ExperimentResult, Folds, ModelConfig, TaskSpec
from ml.tuner import tune_lgbm
from runtime.clock import Clock
from runtime.logger import log
from task.dtypes import feature_schema
from task.metrics import METRICS


@dataclass
class Candidate:
    ds: Dataset
    cfg: ModelConfig
    hypothesis: str
    code: str = ""
    added_cols: list[str] = field(default_factory=list)
    tag: str = ""                 # ключ для «уже пробовали» (например, recipe:freq)


@dataclass
class ResearchState:
    spec: TaskSpec
    folds: Folds
    memory: Memory
    clock: Clock
    tree: SolutionTree
    shifted: list[str] = field(default_factory=list)       # признаки со сдвигом train/test
    results: list[ExperimentResult] = field(default_factory=list)
    analysis: dict[int, dict] = field(default_factory=dict)  # кеш анализа ошибок по id узла

    def analysis_of(self, node: Node) -> dict:
        if node.id not in self.analysis:
            self.analysis[node.id] = error_analysis(node.ds, self.spec, node.result)
        return self.analysis[node.id]


# ---------- работа с признаками ----------

def _full_frame(ds: Dataset) -> pd.DataFrame:
    """Признаки train и test одной таблицей, категории как строки (так проще писать код)."""
    full = pd.concat([ds.X, ds.X_test], ignore_index=True)
    for col in ds.cat_cols:
        full[col] = full[col].astype(object)
    return full


def _with_new_columns(ds: Dataset, new: pd.DataFrame) -> tuple[Dataset | None, list[str]]:
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
        return None, []
    n = len(ds.X)
    ds_new = Dataset(
        X=pd.concat([ds.X, new.iloc[:n].reset_index(drop=True)], axis=1),
        y=ds.y,
        X_test=pd.concat([ds.X_test, new.iloc[n:].reset_index(drop=True)], axis=1),
        test_ids=ds.test_ids,
        cat_cols=cat_cols,
    )
    return ds_new, list(new.columns)


def _top_numeric(parent: Node, k: int) -> list[str]:
    return [c for c in parent.result.feature_importance if c in parent.ds.X.columns
            and c not in parent.ds.cat_cols][:k]


def _recipe_freq(parent: Node, full: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"{c}__freq": full[c].map(full[c].value_counts(normalize=True)).astype(float)
                         for c in parent.ds.cat_cols})


def _recipe_group_diff(parent: Node, full: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in parent.result.feature_importance if c in parent.ds.cat_cols][:2]
    out = {}
    for cat in cats:
        for num in _top_numeric(parent, 3):
            mean = full.groupby(cat)[num].transform("mean")
            out[f"{num}__minus_mean_by_{cat}"] = full[num] - mean
            out[f"{num}__ratio_mean_by_{cat}"] = full[num] / (mean.abs() + 1e-6)
    return pd.DataFrame(out)


def _recipe_ratios(parent: Node, full: pd.DataFrame) -> pd.DataFrame:
    top = _top_numeric(parent, 4)
    return pd.DataFrame({f"{a}__div__{b}": full[a] / (full[b].abs() + 1e-6)
                         for i, a in enumerate(top) for b in top[i + 1:]})


RECIPES: dict[str, tuple[str, Callable]] = {
    "freq": ("Частотное кодирование категорий", _recipe_freq),
    "group_diff": ("Отклонение важных чисел от среднего по важным категориям", _recipe_group_diff),
    "ratios": ("Отношения пар самых важных числовых признаков", _recipe_ratios),
}

# ---------- промпты для Qwen ----------

SEP = (",", ":")   # компактный JSON в промптах — экономим токены

FEATURES_SYSTEM = """Ты — Kaggle Grandmaster по табличным данным. Предложи ОДНУ проверяемую гипотезу новых признаков, которая улучшит метрику, и напиши для неё код.

ВАЖНО: Рассуждай предельно кратко (не более 2-3 предложений). Сразу выводи теги <hypothesis> и <code>.

Как выбирать гипотезу:
1. Начни с анализа ошибок: в каких сегментах или классах модель ошибается сильнее всего и чего ей не хватает, чтобы их различать.
2. Опирайся на смысл колонок и единицы измерения из описания. Сильные признаки обычно доменные: нормировки (на площадь, на время, на объём), отношения и разности связанных величин, отклонение от типичного значения в группе (groupby-transform), агрегаты по сущности, счётчики, флаги аномалий, взаимодействия категорий.
3. Не повторяй идеи из списка «не сработало». Если похожая идея уже отклонена, выбери другое направление.
4. Лучше 3–8 продуманных признаков, чем 30 случайных комбинаций.

Правила кода (нарушение = код упадёт):
- ровно одна функция `def transform_features(df):`; pd и np уже импортированы, другие библиотеки не импортируй;
- df — признаки train и test одной таблицей; таргета в df НЕТ и быть не может;
- верни тот же df с ДОБАВЛЕННЫМИ колонками; исходные колонки не удаляй и не меняй; число строк не меняй;
- используй только колонки из схемы; категориальные колонки — строки (object) с NaN: не делай с ними арифметику, для групп используй df.groupby(col)[num].transform('mean');
- деление только как a / (b.abs() + 1e-6); в конце замени бесконечности: df[new] = df[new].replace([np.inf, -np.inf], np.nan); NaN оставлять можно;
- только векторные операции: без циклов по строкам, без apply(axis=1), без df.append, без inplace=True;
- никаких print, файлов, сети.

Ответ строго в формате:
<hypothesis>1–2 предложения: что добавляешь и почему это должно улучшить метрику</hypothesis>
<code>
def transform_features(df):
    ...
    return df
</code>"""

DEBUG_SYSTEM = """Ты исправляешь упавшую функцию transform_features(df) для pandas.
Сохрани идею признаков, исправь только ошибку. Используй только существующие колонки.
Категориальные колонки — строки (object) с NaN. Таргета в df нет.
Верни только исправленную функцию целиком в теге <code>...</code>, без пояснений."""


def _features_prompt(state: ResearchState, parent: Node) -> str:
    schema = feature_schema(parent.ds, state.spec, parent.result.feature_importance)
    return (f"Задача:\n{state.spec.text[:1200]}\n\n"
            f"Заметки о задаче: {state.spec.notes or '—'}\n\n"
            f"Схема данных (JSON, признаки по убыванию важности):\n{json.dumps(schema, ensure_ascii=False, separators=SEP)}\n\n"
            f"Анализ ошибок текущего решения (OOF):\n{json.dumps(state.analysis_of(parent), ensure_ascii=False, separators=SEP)}\n\n"
            f"Признаки, сильно отличающиеся в train и test (осторожно): {state.shifted}\n\n"
            f"{state.tree.summary(parent)}\n\n"
            f"Метрика {state.spec.metric} ({'больше' if METRICS[state.spec.metric].higher_is_better else 'меньше'} — лучше). "
            f"Предложи следующую гипотезу признаков для решения #{parent.id}.")


def _run_with_debug(code: str, full: pd.DataFrame) -> tuple[pd.DataFrame | None, str, str | None]:
    """Запуск кода; при ошибке LLM чинит его по traceback (до MAX_DEBUG_ATTEMPTS раз)."""
    columns = ", ".join(f"{c}({'str' if full[c].dtype == object else 'num'})" for c in full.columns)
    for attempt in range(config.MAX_DEBUG_ATTEMPTS + 1):
        out, error = run_feature_code(code, full)
        if error is None:
            return out, code, None
        if attempt == config.MAX_DEBUG_ATTEMPTS or not llm.available():
            return None, code, error
        log.info(f"Код признаков упал, LLM чинит по traceback (попытка {attempt + 1}): {error.splitlines()[-1][:150]}")
        fixed = extract_code(llm.ask(DEBUG_SYSTEM, f"Колонки df: {columns}\n\nКод:\n{code}\n\nОшибка:\n{error[-1500:]}",
                                     max_tokens=3000, temperature=0.1))
        if not fixed:
            return None, code, error
        code = fixed
    return None, code, "не удалось исправить"


def _features_llm(state: ResearchState, parent: Node) -> Candidate | None:
    answer = llm.ask(FEATURES_SYSTEM, _features_prompt(state, parent), max_tokens=4000, temperature=0.7)
    code = extract_code(answer)
    if not code:
        return None
    hypothesis = (extract_tag(answer, "hypothesis") or answer.split("<code>")[0]).strip().replace("\n", " ")[:300]
    out, code, error = _run_with_debug(code, _full_frame(parent.ds))
    if error:
        raise RuntimeError(f"Код признаков не исправлен: {error[-300:]}")
    ds, added = _with_new_columns(parent.ds, out)
    if ds is None:
        return None
    return Candidate(ds, parent.cfg.copy(), hypothesis or "Признаки от LLM", code, added)


def _features_heuristic(state: ResearchState, parent: Node) -> Candidate | None:
    for key, (description, recipe) in RECIPES.items():
        tag = f"recipe:{key}"
        if tag in parent.lineage or tag in parent_tried(parent, state):
            continue
        new = recipe(parent, _full_frame(parent.ds))
        ds, added = _with_new_columns(parent.ds, new) if not new.empty else (None, [])
        if ds is not None:
            return Candidate(ds, parent.cfg.copy(), f"[эвристика] {description}", added_cols=added, tag=tag)
    return None


def parent_tried(parent: Node, state: ResearchState) -> set[str]:
    """Что уже пробовали прямо от этого узла (теги детей)."""
    tried: set[str] = set()
    for n in state.tree.nodes.values():
        if n.parent_id == parent.id:
            tried |= (set(n.lineage) - set(parent.lineage)) | {n.action}
    return tried


def features(state: ResearchState, parent: Node) -> Candidate | None:
    if llm.available():
        candidate = _features_llm(state, parent)
        if candidate is not None:
            return candidate
    return _features_heuristic(state, parent)


def merge(state: ResearchState, parent: Node) -> Candidate | None:
    for src in state.tree.merge_sources(parent):
        cols = [c for c in src.added_cols if c in src.ds.X.columns and c not in parent.ds.X.columns]
        if not cols:
            continue
        new = pd.concat([src.ds.X[cols], src.ds.X_test[cols]], ignore_index=True)
        ds, added = _with_new_columns(parent.ds, new)
        if ds is not None:
            return Candidate(ds, parent.cfg.copy(), f"Объединить с удачной веткой #{src.id}: {src.hypothesis[:150]}",
                             added_cols=added, tag=f"merge:{src.id}")
    return None


# ---------- модель и таргет ----------

def target_log(state: ResearchState, parent: Node) -> Candidate:
    cfg = parent.cfg.copy()
    cfg.target_transform = "log1p"
    return Candidate(parent.ds, cfg, "Обучать на log1p(таргета): длинный правый хвост")


def class_balance(state: ResearchState, parent: Node) -> Candidate:
    cfg = parent.cfg.copy()
    cfg.params.setdefault("lgbm", {})["class_weight"] = "balanced"
    cfg.params.setdefault("catboost", {})["auto_class_weights"] = "Balanced"
    return Candidate(parent.ds, cfg, "Балансировка весов классов в моделях: метрика по редким классам")


def tune(state: ResearchState, parent: Node) -> Candidate | None:
    budget = min(config.TUNER_TIME_SEC, 0.25 * state.clock.remaining())
    params = tune_lgbm(parent.ds, state.spec, state.folds, parent.cfg, budget)
    if not params:
        return None
    cfg = parent.cfg.copy()
    cfg.params["lgbm"] = params
    return Candidate(parent.ds, cfg, f"Подобранные Optuna параметры LightGBM: {params}")


def drop_weak(state: ResearchState, parent: Node) -> Candidate | None:
    importance = pd.Series(parent.result.feature_importance)
    share = importance / max(importance.sum(), 1e-12)
    weak = list(share[share < 0.002].index)[: int(0.3 * len(share))]
    if not weak:
        return None
    ds = Dataset(parent.ds.X.drop(columns=weak), parent.ds.y, parent.ds.X_test.drop(columns=weak),
                 parent.ds.test_ids, [c for c in parent.ds.cat_cols if c not in weak])
    return Candidate(ds, parent.cfg.copy(), f"Убрать {len(weak)} слабых признаков: {weak[:10]}")


@dataclass
class Action:
    name: str
    description: str
    run: Callable[[ResearchState, Node], Candidate | None]
    can_run: Callable[[ResearchState, Node], bool]
    repeatable: bool = False      # можно ли применять несколько раз на одном пути


def _label_metric(state: ResearchState) -> bool:
    return state.spec.is_classification and METRICS[state.spec.metric].needs == "label"


ACTIONS: dict[str, Action] = {a.name: a for a in [
    Action("features", "новые признаки (LLM пишет код по анализу ошибок; без LLM — готовые рецепты)", features,
           lambda s, p: llm.available() or any(f"recipe:{k}" not in p.lineage for k in RECIPES), repeatable=True),
    Action("merge", "объединить признаки этого решения с признаками удачной соседней ветки", merge,
           lambda s, p: bool(s.tree.merge_sources(p)), repeatable=True),
    Action("target_log", "обучение на log1p(таргета) для неотрицательного таргета с длинным хвостом", target_log,
           lambda s, p: s.spec.task_type == "regression" and p.cfg.target_transform is None
           and p.ds.y.min() >= 0 and p.ds.y.skew() > 0.5),
    Action("class_balance", "веса классов в моделях (для Macro F1 и других метрик по меткам)", class_balance,
           lambda s, p: _label_metric(s)),
    Action("tune_lgbm", "подбор гиперпараметров LightGBM (Optuna, дорого по времени — лучше на зрелом решении)", tune,
           lambda s, p: "lgbm" in p.cfg.models and s.clock.has(2 * config.TUNER_TIME_SEC)),
    Action("drop_weak", "удалить признаки с почти нулевой важностью", drop_weak,
           lambda s, p: p.ds.X.shape[1] > 10),
]}
