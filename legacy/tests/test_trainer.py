"""
Тесты для модуля кросс-валидационного обучения (ml_core/trainer.py).
"""

import numpy as np
import pytest

from ml_core.trainer import CrossValidatedTrainer, TrainingResult


def test_trainer_regression_cv():
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 4)
    # y = 2*x0 - 3*x1 + шум
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + np.random.normal(0, 0.1, n)
    X_test = np.random.randn(20, 4)

    # 5 фолдов вручную
    indices = np.arange(n)
    folds = [(np.delete(indices, np.arange(i*20, (i+1)*20)), np.arange(i*20, (i+1)*20)) for i in range(5)]

    trainer = CrossValidatedTrainer(task_type="regression", metric="rmse", algorithm="Ridge")
    res: TrainingResult = trainer.train_cv(X_train=X, y_train=y, X_test=X_test, folds=folds)

    assert res.oof_score > 0.0
    assert len(res.oof_preds) == n
    assert len(res.test_preds) == 20
    assert len(res.models) == 5


def test_trainer_multiclass_cv():
    np.random.seed(42)
    n = 90
    X = np.random.randn(n, 3)
    y = np.array(["A", "B", "C"] * 30)
    X_test = np.random.randn(15, 3)

    indices = np.arange(n)
    folds = [(np.delete(indices, np.arange(i*18, (i+1)*18)), np.arange(i*18, (i+1)*18)) for i in range(5)]

    trainer = CrossValidatedTrainer(task_type="multiclass_classification", metric="macro_f1", algorithm="LogisticRegression")
    res: TrainingResult = trainer.train_cv(X_train=X, y_train=y, X_test=X_test, folds=folds)

    assert res.oof_score >= 0.0
    assert len(res.oof_preds) == n
    assert len(res.test_preds) == 15
    assert res.oof_probs is not None
    assert res.oof_probs.shape == (n, 3)
    assert res.test_probs is not None
    assert res.test_probs.shape == (15, 3)
    assert set(res.classes) == {"A", "B", "C"}
