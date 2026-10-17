"""
Агент мета-анализа: выбор модели, функции потерь и метрики.

Отвечает за этап анализа метаданных датасета:
- Загружает промпт с правилами выбора модели из `agent/prompts/01_meta_analysis.txt`.
- Формирует запрос к LLM со структурированными метаданными (`DataProfileReport`).
- Парсит строгий JSON-ответ модели в `ModelSelectionDecision`.
- Содержит детерминированное дерево правил IF-THEN как надежный fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from agent.llm import ask_llm, extract_json
from ml_core.schemas import DataProfileReport, ModelInfo, ModelSelectionDecision

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "01_meta_analysis.txt"


# --------------------------------------------------------------------------- #
# 1. Промпты
# --------------------------------------------------------------------------- #

def get_meta_analysis_system_prompt() -> str:
    """Считывает системный промпт из agent/prompts/01_meta_analysis.txt."""
    if PROMPT_FILE.exists():
        try:
            return PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return (
        "Ты — ведущий ML-архитектор. Проанализируй метаданные датасета и выбери "
        "модель (черновик и боевую), функцию потерь и метрику. Верни СТРОГИЙ JSON."
    )


def build_meta_analysis_prompt(metadata: Union[dict[str, Any], DataProfileReport, str]) -> str:
    """Оборачивает сериализованные метаданные в пользовательский запрос."""
    if isinstance(metadata, DataProfileReport):
        meta_json = metadata.to_json(indent=2)
    elif isinstance(metadata, dict):
        meta_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    else:
        meta_json = str(metadata)

    return (
        "Вот метаданные датасета в формате JSON:\n\n"
        f"```json\n{meta_json}\n```\n\n"
        "На основе дерева решений (IF-THEN) выбери тип задачи, черновую и боевую модель, "
        "функцию потерь и целевую метрику. Верни результат строго в виде валидного JSON."
    )


# --------------------------------------------------------------------------- #
# 2. Детерминированное дерево правил IF - THEN (Fallback)
# --------------------------------------------------------------------------- #

def rule_based_selection(
    metadata: Union[dict[str, Any], DataProfileReport],
) -> ModelSelectionDecision:
    """
    Реализует дерево решений напрямую на Python.

    Правила:
    1. Линейные модели:
       - Явная ось времени (datetime),
       - Размерность признаков > 5000 (раздувание после TF-IDF),
       - Строгая линейная корреляция (>= 0.92).
       -> Черновик: Ridge / LogisticRegression.
       -> Боевая: те же модели с подбором параметров в Optuna.
    2. Градиентный бустинг (стандарт для табличных данных):
       -> Черновик: LightGBM (быстрый расчет, малая глубина, Feature Importances).
       -> Боевая: CatBoost (максимум качества, Optuna).
    3. Легкие нейросети:
       -> Нелинейные данные > 50000 строк -> TabNet / MLP.
    """
    if isinstance(metadata, DataProfileReport):
        meta_dict = metadata.to_dict()
    else:
        meta_dict = dict(metadata)

    target = meta_dict.get("target")
    col_types = meta_dict.get("column_types", {})
    nunique = meta_dict.get("nunique", {})
    head = meta_dict.get("head", [])
    n_rows = meta_dict.get("n_rows", len(head) if head else 1000)

    # Определение задачи
    target_type = col_types.get(target) if target else None
    target_nunique = nunique.get(target, 2) if target else 2

    # Проверка дробности таргета: цель — понять, содержит ли таргет нецелые числа.
    # ВАЖНО: mean и std всегда дробны даже для целочисленного таргета (0,1,2 → mean=1.0, std=0.817).
    # Поэтому проверяем только min и max — если оба целые, сам набор значений тоже целочисленный.
    # Плюс запасная проверка по head-строкам для dict-метаданных без distributions.
    has_fractional = False
    distributions = meta_dict.get("distributions", {})
    if target and target in distributions:
        dist = distributions[target]
        # Только min и max надёжно отражают граничные значения таргета
        boundary_vals = [dist.get("min"), dist.get("max")]
        has_fractional = any(
            isinstance(v, (float, np.floating)) and not float(v).is_integer()
            for v in boundary_vals if v is not None
        )
    if not has_fractional and head:
        # Запасной вариант: проверяем head (может быть доступно при dict-метаданных без distributions)
        target_vals = [row.get(target) for row in head if row.get(target) is not None]
        has_fractional = any(
            isinstance(v, (float, np.floating)) and not float(v).is_integer()
            for v in target_vals
        )


    # Уточнённая логика определения типа задачи:
    # - Бинарная: ровно 2 уникальных значения таргета (независимо от типа колонки).
    # - Мультикласс: категориальный таргет ИЛИ целочисленный numeric с малым nunique (≤ 15).
    #   Порог 15 вместо 20 — снижает риск путать порядковую регрессию с классификацией.
    # - Регрессия: всё остальное (дробный numeric, большое nunique).
    if target_nunique == 2:
        task_type = "binary_classification"
        loss_function = "logloss"
        metric = "roc_auc"
    elif target_type == "category" or (
        target_type in ("numeric", "text")
        and not has_fractional
        and 2 < target_nunique <= 15
    ):
        task_type = "multiclass_classification"
        loss_function = "multiclass"
        metric = "macro_f1"
    else:
        task_type = "regression"
        loss_function = "rmse"
        metric = "rmse"


    # Проверка линейных триггеров
    has_datetime = any(t == "datetime" for t in col_types.values())
    n_features = len(col_types)
    high_corr = False
    for col_info in meta_dict.get("target_correlation", {}).values():
        val = col_info.get("value")
        if val is not None and abs(val) >= 0.92:
            high_corr = True
            break

    is_linear = has_datetime or n_features > 5000 or high_corr
    is_nn = (n_rows > 50000) and not is_linear

    if is_linear:
        if task_type == "regression":
            draft = ModelInfo(
                algorithm="Ridge",
                library_class="sklearn.linear_model.Ridge",
                description="Черновая регрессия Ridge (alpha=1.0) для быстрой оценки весов",
            )
            battle = ModelInfo(
                algorithm="Ridge",
                library_class="sklearn.linear_model.Ridge",
                description="Ridge с подбором регуляризации alpha через Optuna",
                tuning_tool="Optuna",
            )
        else:
            draft = ModelInfo(
                algorithm="LogisticRegression",
                library_class="sklearn.linear_model.LogisticRegression",
                description="Черновая LogisticRegression для моментальной оценки коэффициентов",
            )
            battle = ModelInfo(
                algorithm="LogisticRegression",
                library_class="sklearn.linear_model.LogisticRegression",
                description="LogisticRegression с подбором C и штрафов через Optuna",
                tuning_tool="Optuna",
            )
        reasoning = (
            "Сработал триггер линейной модели: обнаружена ось времени, большая размерность фичей "
            f"или строгая линейная зависимость (datetime={has_datetime}, n_features={n_features}, high_corr={high_corr})."
        )

    elif is_nn:
        draft = ModelInfo(
            algorithm="MLP",
            library_class="torch.nn.Sequential",
            description="Простой 2-слойный MLP для первичной оценки",
        )
        battle = ModelInfo(
            algorithm="TabNet",
            library_class="pytorch_tabnet.tab_model.TabNetClassifier"
            if "class" in task_type
            else "pytorch_tabnet.tab_model.TabNetRegressor",
            description="TabNet для табличных данных с подбором параметров в Optuna",
            tuning_tool="Optuna",
        )
        reasoning = f"Сработал триггер нейросетей: объем выборки > 50000 строк ({n_rows}) и сложные нелинейные зависимости."

    else:
        # Стандарт: Градиентный бустинг
        if task_type == "regression":
            draft = ModelInfo(
                algorithm="LightGBM",
                library_class="lightgbm.LGBMRegressor",
                description="LGBMRegressor с max_depth=3, n_estimators=50 для быстрого Feature Importance",
            )
            battle = ModelInfo(
                algorithm="CatBoost",
                library_class="catboost.CatBoostRegressor",
                description="CatBoostRegressor с обработкой категорий и поиском гиперпараметров в Optuna",
                tuning_tool="Optuna",
            )
        else:
            draft = ModelInfo(
                algorithm="LightGBM",
                library_class="lightgbm.LGBMClassifier",
                description="LGBMClassifier с max_depth=3, n_estimators=50 для быстрого Feature Importance",
            )
            battle = ModelInfo(
                algorithm="CatBoost",
                library_class="catboost.CatBoostClassifier",
                description="CatBoostClassifier с обработкой категорий и поиском гиперпараметров в Optuna",
                tuning_tool="Optuna",
            )
        reasoning = (
            "Выбран градиентный бустинг: разнородные табличные данные (категории, числа, пропуски). "
            "Черновик LightGBM для быстрого Feature Importance, боевая модель CatBoost с Optuna."
        )

    return ModelSelectionDecision(
        task_type=task_type,
        draft_model=draft,
        battle_model=battle,
        loss_function=loss_function,
        metric=metric,
        reasoning=reasoning,
        split_strategy="stratified" if "class" in task_type else ("time_series" if has_datetime else "kfold"),
    )


# --------------------------------------------------------------------------- #
# 3. Выбор модели (LLM с fallback)
# --------------------------------------------------------------------------- #

def select_model_and_metric(
    metadata: Union[dict[str, Any], DataProfileReport],
    ask_llm_fn: Optional[Callable[[str, str], str]] = None,
) -> ModelSelectionDecision:
    """
    Отправляет метаданные в LLM и парсит возвращенный JSON в ModelSelectionDecision.
    При сбое или недоступности модели переключается на правило `rule_based_selection`.
    """
    system_prompt = get_meta_analysis_system_prompt()
    user_prompt = build_meta_analysis_prompt(metadata)

    caller = ask_llm_fn or ask_llm

    try:
        raw_response = caller(system_prompt, user_prompt)
        parsed_json = extract_json(raw_response)
        return ModelSelectionDecision.from_dict(parsed_json)
    except Exception as exc:
        # Fallback на правила
        decision = rule_based_selection(metadata)
        decision.reasoning += f" [Режим fallback: {type(exc).__name__}: {exc}]"
        return decision
