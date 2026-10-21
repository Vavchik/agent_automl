"""
Набор тестов для стадии 2: Мета-анализ и выбор моделей (agent.meta_analyst).

Проверяет:
1. Детерминированное дерево правил IF-THEN (rule_based_selection).
2. Выбор типа задачи (binary, multiclass, regression), лосса и метрики.
3. Триггеры архитектур (Линейные, Градиентный бустинг, Нейросети).
4. Интеграцию с LLM и парсинг строгих JSON-ответов.
5. Отказоустойчивость: переход в fallback-режим при сбоях LLM.
6. Сериализацию структур данных ModelSelectionDecision и ModelInfo.
7. Работу мета-аналитика на реальных датасетах (Titanic, Dry Bean, California Housing, Bike Sharing).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
pd = pytest.importorskip("pandas")

from agent.meta_analyst import (
    build_meta_analysis_prompt,
    get_meta_analysis_system_prompt,
    rule_based_selection,
    select_model_and_metric,
)
from ml_core.data_profiler import profile_data
from ml_core.schemas import DataProfileReport, ModelInfo, ModelSelectionDecision
from ml_core.type_inference import infer_types

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# =========================================================================== #
# 1. Тесты детерминированных правил IF - THEN (rule_based_selection)
# =========================================================================== #

class TestRuleBasedSelection:
    """Тестирование жестких правил выбора архитектур, лосса и метрики."""

    def test_binary_classification_defaults_to_gradient_boosting(self) -> None:
        """Стандартная бинарная классификация на табличных данных -> LightGBM + CatBoost."""
        metadata = {
            "target": "churn",
            "column_types": {"age": "numeric", "department": "category", "churn": "numeric"},
            "nunique": {"age": 50, "department": 4, "churn": 2},
            "n_rows": 1000,
            "target_correlation": {},
        }
        decision = rule_based_selection(metadata)

        assert decision.task_type == "binary_classification"
        assert decision.loss_function == "logloss"
        assert decision.metric == "roc_auc"
        assert decision.split_strategy == "stratified"
        assert decision.draft_model.algorithm == "LightGBM"
        assert decision.battle_model.algorithm == "CatBoost"
        assert decision.battle_model.tuning_tool == "Optuna"

    def test_multiclass_classification(self) -> None:
        """Мультикласс (от 3 до 20 классов) -> multiclass loss, macro_f1."""
        metadata = {
            "target": "species",
            "column_types": {"feat1": "numeric", "feat2": "numeric", "species": "category"},
            "nunique": {"feat1": 100, "feat2": 100, "species": 5},
            "n_rows": 1500,
            "target_correlation": {},
        }
        decision = rule_based_selection(metadata)

        assert decision.task_type == "multiclass_classification"
        assert decision.loss_function == "multiclass"
        assert decision.metric == "macro_f1"
        assert decision.split_strategy == "stratified"
        assert "Classifier" in decision.draft_model.library_class
        assert "Classifier" in decision.battle_model.library_class

    def test_regression_defaults(self) -> None:
        """Непрерывный таргет -> regression, rmse, Regressor."""
        metadata = {
            "target": "price",
            "column_types": {"area": "numeric", "rooms": "numeric", "price": "numeric"},
            "nunique": {"area": 300, "rooms": 5, "price": 450},
            "n_rows": 500,
            "target_correlation": {},
        }
        decision = rule_based_selection(metadata)

        assert decision.task_type == "regression"
        assert decision.loss_function == "rmse"
        assert decision.metric == "rmse"
        assert decision.split_strategy == "kfold"
        assert decision.draft_model.algorithm == "LightGBM"
        assert "Regressor" in decision.draft_model.library_class
        assert decision.battle_model.algorithm == "CatBoost"
        assert "Regressor" in decision.battle_model.library_class

    def test_linear_trigger_datetime_presence(self) -> None:
        """Наличие временной оси (datetime) активирует линейный тренд (Ridge / LogisticRegression)."""
        metadata = {
            "target": "sales",
            "column_types": {"timestamp": "datetime", "promo": "numeric", "sales": "numeric"},
            "nunique": {"timestamp": 365, "promo": 2, "sales": 300},
            "n_rows": 365,
            "target_correlation": {},
        }
        decision = rule_based_selection(metadata)

        assert decision.task_type == "regression"
        assert decision.draft_model.algorithm == "Ridge"
        assert decision.battle_model.algorithm == "Ridge"
        assert decision.split_strategy == "time_series"
        assert "линейной модели" in decision.reasoning

    def test_linear_trigger_high_correlation(self) -> None:
        """Строгая линейная зависимость (корреляция >= 0.92) активирует линейные модели."""
        metadata = {
            "target": "target_col",
            "column_types": {"f1": "numeric", "f2": "numeric", "target_col": "numeric"},
            "nunique": {"f1": 100, "f2": 100, "target_col": 100},
            "n_rows": 200,
            "target_correlation": {
                "f1": {"method": "pearson", "value": 0.95},
                "f2": {"method": "pearson", "value": 0.10},
            },
        }
        decision = rule_based_selection(metadata)

        assert decision.draft_model.algorithm == "Ridge"
        assert "строгая линейная зависимость" in decision.reasoning

    def test_linear_trigger_large_feature_dimension(self) -> None:
        """Раздутая размерность признаков (> 5000 колонок, например после TF-IDF) -> линейная модель."""
        # Генерируем 5005 фичей
        large_cols = {f"tfidf_{i}": "numeric" for i in range(5005)}
        large_cols["target"] = "numeric"
        
        # 1. Для регрессии (target nunique > 20) -> Ridge
        metadata_reg = {
            "target": "target",
            "column_types": large_cols,
            "nunique": {**{c: 10 for c in large_cols}, "target": 500},
            "n_rows": 1000,
            "target_correlation": {},
        }
        decision_reg = rule_based_selection(metadata_reg)
        assert decision_reg.draft_model.algorithm == "Ridge"

        # 2. Для классификации (target nunique == 2) -> LogisticRegression
        metadata_cls = {
            "target": "target",
            "column_types": large_cols,
            "nunique": {**{c: 10 for c in large_cols}, "target": 2},
            "n_rows": 1000,
            "target_correlation": {},
        }
        decision_cls = rule_based_selection(metadata_cls)
        assert decision_cls.draft_model.algorithm == "LogisticRegression"

    def test_neural_network_trigger_large_sample_size(self) -> None:
        """Нелинейные данные большого объема (> 50 000 строк) активируют TabNet / MLP."""
        metadata = {
            "target": "label",
            "column_types": {"num1": "numeric", "num2": "numeric", "label": "numeric"},
            "nunique": {"num1": 10000, "num2": 10000, "label": 2},
            "n_rows": 60000,
            "target_correlation": {},
        }
        decision = rule_based_selection(metadata)

        assert decision.task_type == "binary_classification"
        assert decision.draft_model.algorithm == "MLP"
        assert decision.battle_model.algorithm == "TabNet"
        assert "нейросетей" in decision.reasoning


# =========================================================================== #
# 2. Тесты интеграции с LLM и парсинга (select_model_and_metric)
# =========================================================================== #

class TestLLMIntegrationAndFallback:
    """Тестирование взаимодействия с LLM: успешный парсинг и fallback при ошибках."""

    def test_successful_llm_json_response(self) -> None:
        """Проверка парсинга корректного ответа от LLM."""
        mock_response = json.dumps({
            "task_type": "binary_classification",
            "draft_model": {
                "algorithm": "LightGBM",
                "library_class": "lightgbm.LGBMClassifier",
                "description": "Черновик",
            },
            "battle_model": {
                "algorithm": "CatBoost",
                "library_class": "catboost.CatBoostClassifier",
                "description": "Боевая модель",
                "tuning_tool": "Optuna",
            },
            "loss_function": "logloss",
            "metric": "roc_auc",
            "reasoning": "Рассуждения модели LLM о датасете.",
            "split_strategy": "stratified",
        })

        def mock_llm_fn(sys: str, user: str) -> str:
            return mock_response

        meta = {"target": "churn", "column_types": {"churn": "numeric"}, "nunique": {"churn": 2}}
        decision = select_model_and_metric(meta, ask_llm_fn=mock_llm_fn)

        assert decision.task_type == "binary_classification"
        assert decision.draft_model.algorithm == "LightGBM"
        assert decision.battle_model.algorithm == "CatBoost"
        assert decision.battle_model.tuning_tool == "Optuna"
        assert decision.reasoning == "Рассуждения модели LLM о датасете."

    def test_llm_markdown_code_block_parsing(self) -> None:
        """Проверка парсинга, если LLM обернула JSON в ```json ... ```."""
        mock_response = (
            "Конечно, вот мой выбор архитектуры:\n\n"
            "```json\n"
            "{\n"
            '  "task_type": "regression",\n'
            '  "draft_model": {"algorithm": "LightGBM", "library_class": "lightgbm.LGBMRegressor"},\n'
            '  "battle_model": {"algorithm": "CatBoost", "library_class": "catboost.CatBoostRegressor", "tuning_tool": "Optuna"},\n'
            '  "loss_function": "rmse",\n'
            '  "metric": "r2",\n'
            '  "reasoning": "Табличная регрессия без выбросов"\n'
            "}\n"
            "```\n"
            "Надеюсь, это поможет!"
        )

        def mock_llm_fn(sys: str, user: str) -> str:
            return mock_response

        meta = {"target": "price", "column_types": {"price": "numeric"}, "nunique": {"price": 100}}
        decision = select_model_and_metric(meta, ask_llm_fn=mock_llm_fn)

        assert decision.task_type == "regression"
        assert decision.metric == "r2"
        assert decision.battle_model.algorithm == "CatBoost"

    def test_fallback_on_network_connection_error(self) -> None:
        """Если LLM недоступна (ConnectionError), система не падает, а включает rule_based fallback."""
        def mock_failing_llm(sys: str, user: str) -> str:
            raise ConnectionError("Не удалось подключиться к Ollama/OpenAI API")

        meta = {
            "target": "churn",
            "column_types": {"churn": "numeric", "age": "numeric"},
            "nunique": {"churn": 2, "age": 40},
            "n_rows": 300,
        }
        decision = select_model_and_metric(meta, ask_llm_fn=mock_failing_llm)

        assert decision.task_type == "binary_classification"
        assert decision.draft_model.algorithm == "LightGBM"
        assert "[Режим fallback:" in decision.reasoning
        assert "ConnectionError" in decision.reasoning

    def test_fallback_on_garbage_json(self) -> None:
        """Если LLM вернула невалидный текст вместо JSON, также срабатывает fallback."""
        def mock_garbage_llm(sys: str, user: str) -> str:
            return "Я думаю, что тут лучше всего подойдет Случайный Лес, но JSON я вам не дам!"

        meta = {
            "target": "target",
            "column_types": {"target": "numeric"},
            "nunique": {"target": 100},
        }
        decision = select_model_and_metric(meta, ask_llm_fn=mock_garbage_llm)

        assert decision.task_type == "regression"
        assert "[Режим fallback:" in decision.reasoning


# =========================================================================== #
# 3. Тесты промптов и сериализации схем (ModelSelectionDecision)
# =========================================================================== #

class TestPromptAndSchemas:
    """Тестирование формирования промптов и корректности сериализации схем данных."""

    def test_system_prompt_loading(self) -> None:
        """Проверка, что системный промпт считывается из 01_meta_analysis.txt."""
        prompt = get_meta_analysis_system_prompt()
        assert "ML-архитектор" in prompt or "Senior Data Scientist" in prompt
        assert "ЧЕРНОВИК" in prompt or "Draft" in prompt

    def test_build_meta_analysis_prompt(self) -> None:
        """Проверка оборачивания словаря в промпт для LLM."""
        meta = {"target": "is_fraud", "n_rows": 500}
        prompt = build_meta_analysis_prompt(meta)
        assert "```json" in prompt
        assert '"target": "is_fraud"' in prompt

    def test_model_selection_decision_serialization(self) -> None:
        """Проверка round-trip сериализации to_dict() и from_dict()."""
        original = ModelSelectionDecision(
            task_type="binary_classification",
            draft_model=ModelInfo(algorithm="Ridge", library_class="sklearn.linear_model.Ridge"),
            battle_model=ModelInfo(algorithm="CatBoost", library_class="catboost.CatBoostClassifier", tuning_tool="Optuna"),
            loss_function="logloss",
            metric="roc_auc",
            reasoning="Тестовое обоснование",
            split_strategy="stratified",
        )

        d = original.to_dict()
        assert d["task_type"] == "binary_classification"
        assert d["draft_model"]["algorithm"] == "Ridge"

        restored = ModelSelectionDecision.from_dict(d)
        assert restored.task_type == original.task_type
        assert restored.draft_model.algorithm == original.draft_model.algorithm
        assert restored.battle_model.tuning_tool == "Optuna"


# =========================================================================== #
# 4. Тесты на реальных датасетах из data/raw/
# =========================================================================== #

class TestRealDatasetsMetaAnalysis:
    """Проверка работы мета-аналитика на реальных бенчмарк-датасетах."""

    @pytest.mark.skipif(not (DATA_RAW_DIR / "titanic.csv").exists(), reason="titanic.csv не найден")
    def test_titanic_meta_analysis(self) -> None:
        """Titanic: бинарная классификация (survived) -> LightGBM + CatBoost, logloss, roc_auc."""
        df = pd.read_csv(DATA_RAW_DIR / "titanic.csv")
        clean_df, types = infer_types(df)
        target = "survived" if "survived" in types else "Survived"

        report = profile_data(clean_df, types, target=target)
        decision = rule_based_selection(report)

        assert decision.task_type == "binary_classification"
        assert decision.loss_function == "logloss"
        assert decision.metric == "roc_auc"
        assert decision.split_strategy == "stratified"
        assert decision.draft_model.algorithm == "LightGBM"
        assert decision.battle_model.algorithm == "CatBoost"

    def test_multiclass_benchmark_meta_analysis(self) -> None:
        """Мультиклассовый бенчмарк (на примере load_wine) -> multiclass, macro_f1, LightGBM + CatBoost."""
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame.copy()

        clean_df, types = infer_types(df)
        report = profile_data(clean_df, types, target="target")
        decision = rule_based_selection(report)

        assert decision.task_type == "multiclass_classification"
        assert decision.loss_function == "multiclass"
        assert decision.metric == "macro_f1"
        assert decision.split_strategy == "stratified"
        assert decision.draft_model.algorithm == "LightGBM"
        assert decision.battle_model.algorithm == "CatBoost"

    def test_iris_triggers_linear_model_due_to_high_correlation(self) -> None:
        """В Iris лепестки имеют корреляцию > 0.92 с классом -> триггер линейной модели (LogisticRegression)."""
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame.copy()

        clean_df, types = infer_types(df)
        report = profile_data(clean_df, types, target="target")
        decision = rule_based_selection(report)

        assert decision.task_type == "multiclass_classification"
        assert decision.draft_model.algorithm == "LogisticRegression"
        assert "строгая линейная зависимость" in decision.reasoning

    @pytest.mark.skipif(not (DATA_RAW_DIR / "california_housing.csv").exists(), reason="california_housing.csv не найден")
    def test_california_housing_regression_meta_analysis(self) -> None:
        """California Housing: регрессия (MedHouseVal) -> rmse, rmse."""
        df = pd.read_csv(DATA_RAW_DIR / "california_housing.csv")
        clean_df, types = infer_types(df)
        target = "MedHouseVal"

        report = profile_data(clean_df, types, target=target)
        decision = rule_based_selection(report)

        assert decision.task_type == "regression"
        assert decision.loss_function == "rmse"
        assert decision.metric == "rmse"
        assert decision.draft_model.algorithm == "LightGBM"

    @pytest.mark.skipif(not (DATA_RAW_DIR / "bike_sharing.csv").exists(), reason="bike_sharing.csv не найден")
    def test_bike_sharing_temporal_meta_analysis(self) -> None:
        """Bike Sharing: регрессия с осью времени datetime -> линейный триггер / time_series."""
        df = pd.read_csv(DATA_RAW_DIR / "bike_sharing.csv")
        # Создаем явную строковую колонку даты для проверки инференса даты и временного триггера
        df["date_str"] = "2023-01-01"
        clean_df, types = infer_types(df)
        target = "count" if "count" in types else ("cnt" if "cnt" in types else df.columns[-2])

        report = profile_data(clean_df, types, target=target)
        decision = rule_based_selection(report)

        # Присутствует колонка типа datetime -> time_series и Ridge
        assert decision.split_strategy == "time_series"
        assert decision.draft_model.algorithm == "Ridge"

    def test_live_local_qwen_meta_analysis(self) -> None:
        """Тест реального вызова локальной модели Qwen через Ollama (если сервис запущен)."""
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as resp:
                is_online = (resp.status == 200)
        except Exception:
            is_online = False

        if not is_online:
            pytest.skip("Локальный сервис Ollama не доступен на http://localhost:11434")

        metadata = {
            "target": "survived",
            "column_types": {"age": "numeric", "sex": "category", "survived": "numeric"},
            "nunique": {"age": 50, "sex": 2, "survived": 2},
            "n_rows": 891,
            "target_correlation": {"sex": {"method": "cramers_v", "value": 0.54}},
        }
        decision = select_model_and_metric(metadata)

        assert "class" in decision.task_type.lower()
        assert decision.metric in ("roc_auc", "pr_auc", "accuracy", "f1")
        assert decision.draft_model.algorithm is not None
        assert decision.battle_model.algorithm is not None


