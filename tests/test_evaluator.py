"""
Тесты для модуля оценки моделей (ml_core/evaluator.py) и критериев эскалации (EvalResult).

Проверяет:
1. Логику EvalResult:
   - is_better_than_dummy (превосходство над наивным бейзлайном);
   - is_overfit (детектирование переобучения по train/val gap > 15%);
   - tier1_passed (комплексный флаг успешности Tier 1);
2. Функцию evaluate_model():
   - Бинарная классификация (ROC-AUC, Accuracy);
   - Мультиклассовая классификация (Macro-F1);
   - Регрессия (RMSE, MAE, R2);
   - Dummy-бейзлайн (DummyClassifier, DummyRegressor);
3. Интеграцию с LangGraph и условные переходы (Conditional Edges):
   - Если Tier 1 успешен -> переход в END;
   - Если Tier 1 провалился (низкий скор / оверфит) -> эскалация в Tier 2 (automl);
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from agent.graph import app as pipeline_graph, make_initial_state, should_escalate_to_tier2, should_escalate_to_tier3
from ml_core.evaluator import compute_dummy_score, compute_metric, evaluate_model
from ml_core.schemas import EvalResult, SplitResult


# --------------------------------------------------------------------------- #
# 1. Тесты свойств EvalResult
# --------------------------------------------------------------------------- #

class TestEvalResultLogic:
    def test_classification_tier1_passed(self) -> None:
        """Хорошая модель классификации: выше dummy, без оверфита, выше порога."""
        res = EvalResult(
            val_score=0.88,
            train_score=0.91,
            dummy_score=0.50,
            metric_name="roc_auc",
            threshold=0.80,
        )
        assert res.is_better_than_dummy is True
        assert res.is_overfit is False
        assert res.tier1_passed is True

    def test_classification_overfit_detected(self) -> None:
        """Переобучение: разрыв train и val > 15%."""
        res = EvalResult(
            val_score=0.72,
            train_score=0.95,  # 0.95 - 0.72 = 0.23 > 0.15
            dummy_score=0.50,
            metric_name="roc_auc",
            threshold=0.70,
        )
        assert res.is_better_than_dummy is True
        assert res.is_overfit is True
        assert res.tier1_passed is False  # Оверфит блокирует Tier 1

    def test_classification_not_better_than_dummy(self) -> None:
        """Модель практически на уровне случайного угадывания (разрыв <= 5%)."""
        res = EvalResult(
            val_score=0.53,
            train_score=0.54,
            dummy_score=0.50,  # 0.53 - 0.50 = 0.03 <= 0.05
            metric_name="roc_auc",
            threshold=0.0,
        )
        assert res.is_better_than_dummy is False
        assert res.tier1_passed is False

    def test_classification_below_user_threshold(self) -> None:
        """Модель лучше бейзлайна, но не дотянула до строгого порога пользователя."""
        res = EvalResult(
            val_score=0.78,
            train_score=0.80,
            dummy_score=0.50,
            metric_name="roc_auc",
            threshold=0.85,  # Порог 0.85 не достигнут
        )
        assert res.is_better_than_dummy is True
        assert res.is_overfit is False
        assert res.tier1_passed is False

    def test_regression_tier1_passed(self) -> None:
        """Регрессия (RMSE): меньше = лучше, наша модель значительно точнее dummy."""
        res = EvalResult(
            val_score=10.0,
            train_score=9.5,
            dummy_score=25.0,  # 10.0 < 25.0 * 0.95 (23.75)
            metric_name="rmse",
            threshold=15.0,
        )
        assert res.is_better_than_dummy is True
        assert res.is_overfit is False
        assert res.tier1_passed is True

    def test_regression_overfit_detected(self) -> None:
        """Регрессия: на train RMSE=2.0, на val RMSE=10.0 (сильный переобув)."""
        res = EvalResult(
            val_score=10.0,
            train_score=2.0,  # gap > 15%
            dummy_score=25.0,
            metric_name="rmse",
            threshold=0.0,
        )
        assert res.is_overfit is True
        assert res.tier1_passed is False


# --------------------------------------------------------------------------- #
# 2. Тесты функции evaluate_model()
# --------------------------------------------------------------------------- #

class TestEvaluateModelFunction:
    def test_evaluate_binary_logistic_regression(self) -> None:
        """Оценка реальной модели логистической регрессии на синтетических данных."""
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4, random_state=42
        )
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        eval_res = evaluate_model(
            model=clf,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metric="roc_auc",
            threshold=0.70,
        )

        assert eval_res.metric_name == "roc_auc"
        assert eval_res.dummy_score == 0.5
        assert eval_res.val_score > 0.75
        assert eval_res.is_better_than_dummy is True
        assert eval_res.tier1_passed is True

    def test_evaluate_regression_ridge(self) -> None:
        """Оценка модели Ridge регрессии."""
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        reg = Ridge()
        reg.fit(X_train, y_train)

        eval_res = evaluate_model(
            model=reg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metric="rmse",
            threshold=5.0,
        )

        assert eval_res.metric_name == "rmse"
        assert eval_res.val_score < eval_res.dummy_score
        assert eval_res.is_better_than_dummy is True
        assert eval_res.tier1_passed is True

    def test_compute_metric_macro_f1(self) -> None:
        """Проверка расчета Macro-F1 для мультикласса."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        score = compute_metric(y_true, y_pred, metric="macro_f1")
        assert pytest.approx(score, 0.001) == 1.0


# --------------------------------------------------------------------------- #
# 3. Тесты эскалации и маршрутизации графа (LangGraph)
# --------------------------------------------------------------------------- #

class TestGraphEscalationRouting:
    def test_router_tier1_to_done(self) -> None:
        """Когда Tier 1 успешен, маршрутизатор возвращает 'done' (переход в END)."""
        good_eval = EvalResult(
            val_score=0.90,
            train_score=0.92,
            dummy_score=0.50,
            metric_name="roc_auc",
            threshold=0.80,
        )
        state = {"eval_result": good_eval, "trained_model": "mock_model"}
        next_node = should_escalate_to_tier2(state)
        assert next_node == "done"

    def test_router_tier1_to_automl_on_failure(self) -> None:
        """Когда Tier 1 провалился (низкий скор / оверфит), маршрутизатор эскалирует в 'automl'."""
        bad_eval = EvalResult(
            val_score=0.52,
            train_score=0.95,  # Жуткий оверфит
            dummy_score=0.50,
            metric_name="roc_auc",
            threshold=0.80,
        )
        state = {"eval_result": bad_eval, "trained_model": "mock_model"}
        next_node = should_escalate_to_tier2(state)
        assert next_node == "automl"

    def test_router_tier2_to_agent_coder(self) -> None:
        """Когда AutoML не дал нужного скора, маршрутизатор эскалирует в 'agent_coder'."""
        state = {
            "automl_score": 0.60,
            "target_metric_threshold": 0.85,
            "eval_result": None,
        }
        next_node = should_escalate_to_tier3(state)
        assert next_node == "agent_coder"

    def test_end_to_end_graph_with_trained_model_and_escalation(self) -> None:
        """
        Сквозной прогон графа с передачей обученной модели:
        - Имитируем провальную модель с высоким порогом (threshold=0.99);
        - Граф должен выполнить profiling -> meta_analysis -> splitting -> evaluate
          и через conditional edge эскалировать в automl -> agent_coder -> END.
        """
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

        class FailingModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                # Равные вероятности (AUC = 0.50, на уровне Dummy)
                p = np.full((len(X), 2), 0.5)
                return p

        weak_model = FailingModel()

        state = make_initial_state(
            df=df,
            target="target",
            test_size=0.3,
            val_size=0.0,
            target_metric_threshold=0.85,
        )
        state["trained_model"] = weak_model

        final_state = pipeline_graph.invoke(state)

        # Проверяем, что граф произвел оценку
        assert final_state["eval_result"] is not None
        assert final_state["eval_result"].tier1_passed is False
        # Проверяем, что граф эскалировал до Tier 3
        assert final_state["tier"] == 3
