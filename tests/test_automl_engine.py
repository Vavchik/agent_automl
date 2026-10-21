"""
Тесты для модуля тяжелого ансамблирования (ml_core/automl_engine.py).
"""

from ml_core.automl_engine import AutoMLEnsembleEngine, METRIC_MAPPING


def test_automl_engine_initialization_and_fallback():
    engine = AutoMLEnsembleEngine(
        task_type="multiclass_classification",
        metric="macro_f1",
        time_limit_sec=60,
    )
    assert engine.ag_metric == "f1_macro"
    assert engine.is_classification is True

    # Check fallback when autogluon is not available or train_df is None
    score, preds = engine.fit_and_predict(None, None, target_col="target")
    assert score is None
    assert preds is None


def test_metric_mapping_regression():
    engine = AutoMLEnsembleEngine(
        task_type="regression",
        metric="rmse",
        time_limit_sec=60,
    )
    assert engine.ag_metric == "root_mean_squared_error"
    assert engine.is_classification is False
