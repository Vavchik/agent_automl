"""
Unit tests for agent/meta_analyst.py (pure Python tests without pandas dependency).
"""

from agent.meta_analyst import (
    build_meta_analysis_prompt,
    rule_based_selection,
    select_model_and_metric,
)


def test_rule_based_selection_hints():
    # 1. Проверка подсказки регрессии
    meta = {
        "target": "site_eui",
        "column_types": {"feat1": "numeric", "site_eui": "numeric"},
        "nunique": {"site_eui": 1000},
        "n_rows": 5000,
    }
    decision = rule_based_selection(meta, task_type_hint="regression", metric_hint="rmse")
    assert decision.task_type == "regression"
    assert decision.metric == "rmse"
    assert decision.loss_function == "rmse"

    # 2. Проверка подсказки мультикласса и Macro F1
    meta_cls = {
        "target": "attack_cat",
        "column_types": {"dur": "numeric", "attack_cat": "category"},
        "nunique": {"attack_cat": 10},
        "n_rows": 10000,
    }
    decision_cls = rule_based_selection(
        meta_cls,
        task_type_hint="multiclass_classification",
        metric_hint="macro_f1",
    )
    assert decision_cls.task_type == "multiclass_classification"
    assert decision_cls.metric == "macro_f1"
    assert decision_cls.loss_function == "multiclass"
    assert decision_cls.split_strategy == "stratified"


def test_build_meta_analysis_prompt_with_context():
    prompt = build_meta_analysis_prompt(
        metadata={"target": "site_eui", "n_rows": 500},
        task_description="Определение удельного энергопотребления здания",
        metric_hint="RMSE",
    )
    assert "## Описание задачи из task.md:" in prompt
    assert "Определение удельного энергопотребления здания" in prompt
    assert "## Требуемая метрика оценки (из условия):" in prompt
    assert "RMSE" in prompt
    assert "## Метаданные датасета (JSON):" in prompt


def test_select_model_and_metric_fallback_with_hints():
    def failing_llm(sys_prompt, usr_prompt):
        raise ConnectionError("LLM server offline")

    decision = select_model_and_metric(
        metadata={"target": "site_eui", "nunique": {"site_eui": 500}, "n_rows": 2000},
        task_description="Регрессия энергопотребления",
        metric_hint="rmse",
        task_type_hint="regression",
        ask_llm_fn=failing_llm,
    )
    assert decision.task_type == "regression"
    assert decision.metric == "rmse"
    assert "Режим fallback" in decision.reasoning
