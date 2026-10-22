"""
Граф пайплайна AutoML на базе LangGraph с 3-уровневой каскадной архитектурой.

Архитектура каскада (Escalation Pipeline):
- Tier 1: Наш жесткий ML-пайплайн (LightGBM/CatBoost + Optuna).
  1. Узел «Профилирование» (Жесткий код)
  2. Узел «Мета-анализ» (АГЕНТ: LLM Qwen выбор модели и метрики)
  3. Узел «Разбиение данных» (Жесткий код: train/val/test)
  4. Узел «Оценка качества» (Жесткий код: Dummy vs Model vs Overfit)
- Tier 2 (Эскалация 1): AutoML библиотека (FLAML с time_budget).
- Tier 3 (Эскалация 2): LLM-агент (Error Analysis + Feature Engineering + Code Gen).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict
import pandas as pd
from langgraph.graph import END, StateGraph

from agent.meta_analyst import select_model_and_metric
from ml_core.data_profiler import profile_data
from ml_core.evaluator import evaluate_model
from ml_core.schemas import DataProfileReport, EvalResult, ModelSelectionDecision, SplitResult
from ml_core.splitter import split_data
from ml_core.type_inference import infer_types


# --------------------------------------------------------------------------- #
# 1. Состояние графа (State)
# --------------------------------------------------------------------------- #

class PipelineState(TypedDict, total=False):
    # Входные данные
    df: pd.DataFrame
    target: str
    time_col: Optional[str]
    group_col: Optional[str]
    test_size: float
    val_size: float
    target_metric_threshold: float

    # Результат Этапа 1 (Жесткая нода: типы и профиль)
    clean_df: pd.DataFrame
    column_types: dict[str, str]
    report: DataProfileReport

    # Результат Этапа 2 (АГЕНТ: выбор модели и метрики)
    decision: ModelSelectionDecision

    # Результат Этапа 3 (Жесткая нода: train/val/test сплиты)
    splits: SplitResult

    # Stage 4-5: Drafter + Tuner + Trainer
    draft_score: Optional[float]          # Score черновика на val
    important_features: list[str]         # Признаки после prune
    best_params: dict[str, Any]           # Лучшие параметры Optuna
    trained_model: Any                    # Обученная боевая модель

    # Stage 6: Evaluator
    eval_result: Optional[EvalResult]     # Структура с метриками и флагами
    tier: int                             # Текущий активный tier (1/2/3)

    # Stage 7: AutoML
    automl_model: Any                     # Модель от FLAML
    automl_score: Optional[float]         # Score AutoML на val

    # Stage 9: LLM Coder
    error_analysis: Optional[str]         # Анализ ошибок моделей Tier 1 и 2
    generated_code: Optional[str]         # Python-код, сгенерированный агентом
    agent_score: Optional[float]          # Score решения агента

    # Служебные поля
    error: Optional[str]


# Псевдоним для обратной совместимости
AgentState = PipelineState


def make_initial_state(
    df: pd.DataFrame,
    target: str,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    target_metric_threshold: float = 0.0,
) -> PipelineState:
    """Создает стартовое состояние для запуска графа."""
    return PipelineState(
        df=df,
        target=target,
        time_col=time_col,
        group_col=group_col,
        test_size=test_size,
        val_size=val_size,
        target_metric_threshold=target_metric_threshold,
        tier=1,
        eval_result=None,
        draft_score=None,
        important_features=[],
        best_params={},
        trained_model=None,
        automl_model=None,
        automl_score=None,
        error_analysis=None,
        generated_code=None,
        agent_score=None,
        error=None,
    )


# --------------------------------------------------------------------------- #
# 2. Узлы графа
# --------------------------------------------------------------------------- #

def profiling_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 1 (Жесткий код):
    Очистка типов и сбор математического профиля данных без участия LLM.
    """
    df = state["df"]
    target = state["target"]

    clean_df, column_types = infer_types(df)
    report = profile_data(clean_df, column_types, target=target)

    return {
        "clean_df": clean_df,
        "column_types": column_types,
        "report": report,
    }


def meta_analysis_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 2 (АГЕНТ: LLM Qwen):
    ЕДИНСТВЕННЫЙ ЭТАП, где работает агент!
    Анализирует метаданные профиля и выбирает архитектуру, функцию потерь и метрику.
    """
    report = state["report"]
    decision = select_model_and_metric(report)
    return {"decision": decision}


def splitting_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 3 (Жесткий код):
    Детерминированное разбиение выборки на train / val / test без участия LLM.
    """
    clean_df = state["clean_df"]
    target = state["target"]
    decision = state.get("decision")
    strategy = decision.split_strategy if decision else None

    splits = split_data(
        df=clean_df,
        target=target,
        strategy=strategy,
        test_size=state.get("test_size", 0.2),
        val_size=state.get("val_size", 0.1),
        time_col=state.get("time_col"),
        group_col=state.get("group_col"),
    )
    return {"splits": splits}


def evaluate_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 6 (Жесткий код: Оценка Tier 1):
    Замеряет качество обученной модели, обучает Dummy-бейзлайн и проверяет оверфиттинг.
    """
    model = state.get("trained_model")
    splits = state.get("splits")
    decision = state.get("decision")

    # Если модель еще не обучена в графе (этапы 4-5 в разработке), пропускаем оценку
    if model is None or splits is None:
        return {"eval_result": None, "tier": 1}

    target = state["target"]
    metric = decision.metric if decision else "roc_auc"
    threshold = state.get("target_metric_threshold", 0.0)

    # Выделяем train и validation выборки
    X_train = splits.train.drop(columns=[target])
    y_train = splits.train[target]

    val_df = splits.val if (splits.val is not None and len(splits.val) > 0) else splits.test
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    eval_result = evaluate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric=metric,
        threshold=threshold,
    )

    return {"eval_result": eval_result, "tier": 1}


def automl_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 7 (Жесткий код: Tier 2 AutoML FLAML):
    Запускает FLAML AutoML с жестким time_budget при неудаче Tier 1.
    """
    # До реализации шага 6 (ml_core/automl_runner.py) фиксируем переход на Tier 2
    return {"tier": 2}


def agent_coder_node(state: PipelineState) -> dict[str, Any]:
    """
    Узел 9 (АГЕНТ: Tier 3 LLM Coder):
    Итеративный агент для анализа ошибок и Feature Engineering при неудаче Tier 1 и 2.
    """
    # До реализации шага 7 (agent/coder_agent.py) фиксируем переход на Tier 3
    return {"tier": 3}


# --------------------------------------------------------------------------- #
# 3. Маршрутизаторы условных переходов (Conditional Edges)
# --------------------------------------------------------------------------- #

def should_escalate_to_tier2(state: PipelineState) -> str:
    """
    Роутер Tier 1 -> Tier 2:
    - Если модель удовлетворяет порогу качества и нет оверфита -> 'done' (END);
    - Если модель не передана (ранние стадии пайплайна) -> 'done';
    - Иначе эскалация -> 'automl' (Tier 2).
    """
    eval_result = state.get("eval_result")
    if eval_result is not None:
        if eval_result.tier1_passed:
            return "done"
        return "automl"

    # Если модели не было (например, запускаем только профилирование и сплит)
    if state.get("trained_model") is None:
        return "done"

    return "automl"


def should_escalate_to_tier3(state: PipelineState) -> str:
    """
    Роутер Tier 2 -> Tier 3:
    - Если AutoML успешно справился с задачей -> 'done' (END);
    - Иначе эскалация к независимому LLM-агенту кодеру -> 'agent_coder' (Tier 3).
    """
    automl_score = state.get("automl_score")
    eval_result = state.get("eval_result")
    threshold = state.get("target_metric_threshold", 0.0)

    if automl_score is not None:
        if threshold > 0.0 and automl_score >= threshold:
            return "done"
        if eval_result and automl_score > eval_result.val_score:
            return "done"

    # Если модуль FLAML еще в разработке или AutoML не улучшил качество -> Tier 3
    return "agent_coder"


# --------------------------------------------------------------------------- #
# 4. Сборка графа
# --------------------------------------------------------------------------- #

workflow = StateGraph(PipelineState)

# Добавляем узлы
workflow.add_node("profiling", profiling_node)          # Узел 1: Чистый код
workflow.add_node("meta_analysis", meta_analysis_node)  # Узел 2: АГЕНТ (LLM Qwen)
workflow.add_node("splitting", splitting_node)          # Узел 3: Чистый код
workflow.add_node("evaluate", evaluate_node)            # Узел 6: Оценка Tier 1
workflow.add_node("automl", automl_node)                # Узел 7: Tier 2 (FLAML)
workflow.add_node("agent_coder", agent_coder_node)      # Узел 9: Tier 3 (LLM Coder)

# Входная точка и линейная часть до оценки
workflow.set_entry_point("profiling")
workflow.add_edge("profiling", "meta_analysis")
workflow.add_edge("meta_analysis", "splitting")
workflow.add_edge("splitting", "evaluate")

# Условные переходы каскада
workflow.add_conditional_edges(
    "evaluate",
    should_escalate_to_tier2,
    {
        "done": END,
        "automl": "automl",
    },
)

workflow.add_conditional_edges(
    "automl",
    should_escalate_to_tier3,
    {
        "done": END,
        "agent_coder": "agent_coder",
    },
)

workflow.add_edge("agent_coder", END)

# Компиляция исполняемого графа
app = workflow.compile()
