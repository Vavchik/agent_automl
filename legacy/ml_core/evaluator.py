"""
Стадия 6 / 8 / 10: Оценка качества моделей и расчет критериев эскалации.

Модуль отвечает за:
1. Расчет целевых метрик (ROC-AUC, Macro-F1, Accuracy, RMSE, MAE, R2).
2. Обучение наивного бейзлайна (DummyClassifier / DummyRegressor).
3. Проверку условий эскалации (is_better_than_dummy, is_overfit, tier1_passed).
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    DummyClassifier = DummyRegressor = None

from ml_core.schemas import EvalResult


CLASSIFICATION_METRICS = {"roc_auc", "macro_f1", "f1_macro", "weighted_f1", "accuracy", "logloss", "multiclass"}


def _is_classification_metric(metric: str) -> bool:
    return metric.lower() in CLASSIFICATION_METRICS


def _numpy_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Чистый numpy расчет Macro F1 без зависимости от scikit-learn."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 0:
        return 0.0
    f1_list = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1_list.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1_list))


def compute_metric(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
    metric: str = "roc_auc",
) -> float:
    """
    Вычисляет значение указанной метрики качества (через sklearn или numpy fallback).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metric_norm = metric.lower()

    if metric_norm in ("macro_f1", "f1_macro", "f1"):
        if HAS_SKLEARN:
            return float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        return _numpy_macro_f1(y_true_arr, y_pred_arr)

    if metric_norm in ("weighted_f1", "f1_weighted"):
        if HAS_SKLEARN:
            return float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        return _numpy_macro_f1(y_true_arr, y_pred_arr)

    if metric_norm == "accuracy":
        if HAS_SKLEARN:
            return float(accuracy_score(y_true_arr, y_pred_arr))
        return float(np.mean(y_true_arr == y_pred_arr))

    if metric_norm in ("rmse", "root_mean_squared_error"):
        if HAS_SKLEARN:
            mse = mean_squared_error(y_true_arr, y_pred_arr)
            return float(np.sqrt(mse))
        return float(np.sqrt(np.mean((y_true_arr.astype(float) - y_pred_arr.astype(float)) ** 2)))

    if metric_norm in ("mae", "mean_absolute_error"):
        if HAS_SKLEARN:
            return float(mean_absolute_error(y_true_arr, y_pred_arr))
        return float(np.mean(np.abs(y_true_arr.astype(float) - y_pred_arr.astype(float))))

    if metric_norm in ("r2", "r2_score") and HAS_SKLEARN:
        return float(r2_score(y_true_arr, y_pred_arr))

    if metric_norm in ("logloss", "multiclass") and HAS_SKLEARN:
        if y_proba is not None:
            try:
                return float(log_loss(y_true_arr, y_proba))
            except Exception:
                pass
        return float(log_loss(y_true_arr, y_pred_arr))

    if metric_norm in ("roc_auc", "auc") and HAS_SKLEARN:
        if y_proba is not None:
            proba_arr = np.asarray(y_proba)
            unique_classes = np.unique(y_true_arr)
            if len(unique_classes) < 2:
                return 0.5
            if len(unique_classes) == 2:
                p = proba_arr[:, 1] if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2 else proba_arr.ravel()
                try:
                    return float(roc_auc_score(y_true_arr, p))
                except Exception:
                    pass
            else:
                try:
                    return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
                except Exception:
                    pass
        try:
            return float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            return float(accuracy_score(y_true_arr, y_pred_arr))

    # Fallback
    try:
        return float(np.sqrt(np.mean((y_true_arr.astype(float) - y_pred_arr.astype(float)) ** 2)))
    except Exception:
        return float(np.mean(y_true_arr == y_pred_arr))


def is_metric_better(
    new_score: float,
    base_score: float,
    metric: str = "macro_f1",
    min_delta: float = 0.0,
) -> bool:
    """
    Определяет, стал ли новый скор лучше базового с учетом направления метрики:
    - Для RMSE / MAE / LogLoss: меньше = лучше (new < base - min_delta);
    - Для Macro F1 / ROC-AUC / Accuracy / R2: больше = лучше (new > base + min_delta).
    """
    metric_norm = metric.lower()
    if metric_norm in ("rmse", "root_mean_squared_error", "mae", "mean_absolute_error", "logloss"):
        return new_score < (base_score - min_delta)
    return new_score > (base_score + min_delta)


def evaluate_oof(
    y_true: Union[pd.Series, np.ndarray, list],
    y_oof_pred: Union[pd.Series, np.ndarray, list],
    y_oof_proba: Optional[Union[pd.Series, np.ndarray]] = None,
    metric: str = "macro_f1",
) -> float:
    """
    Вычисляет честную Out-of-Fold (OOF) метрику по всей выборке кросс-валидации.
    """
    return compute_metric(y_true=y_true, y_pred=y_oof_pred, y_proba=y_oof_proba, metric=metric)


def _predict_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    is_cls: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Получает предсказания классов/значений и вероятностей (если доступны)."""
    y_pred = model.predict(X)

    y_proba = None
    if is_cls and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None

    return np.asarray(y_pred), y_proba


def compute_dummy_score(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    metric: str,
) -> float:
    """Обучает наивный бейзлайн (Dummy) на train и замеряет метрику на val."""
    is_cls = _is_classification_metric(metric)

    if is_cls:
        # Для ROC-AUC теоретический случайный/наивный бейзлайн равен 0.5
        if metric.lower() in ("roc_auc", "auc"):
            return 0.5

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        pred, proba = _predict_model(dummy, X_val, is_cls=True)
        return compute_metric(y_val, pred, proba, metric=metric)

    else:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        pred, _ = _predict_model(dummy, X_val, is_cls=False)
        return compute_metric(y_val, pred, None, metric=metric)


def evaluate_model(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    metric: str = "roc_auc",
    threshold: float = 0.0,
) -> EvalResult:
    """
    Комплексная оценка модели:
    1. Считает train_score и val_score.
    2. Обучает Dummy-бейзлайн и считает dummy_score на val.
    3. Возвращает EvalResult с флагами tier1_passed, is_better_than_dummy, is_overfit.
    """
    is_cls = _is_classification_metric(metric)

    # 1. Предсказания на train и val
    train_pred, train_proba = _predict_model(model, X_train, is_cls=is_cls)
    val_pred, val_proba = _predict_model(model, X_val, is_cls=is_cls)

    train_score = compute_metric(y_train, train_pred, train_proba, metric=metric)
    val_score = compute_metric(y_val, val_pred, val_proba, metric=metric)

    # 2. Наивный бейзлайн
    dummy_score = compute_dummy_score(X_train, y_train, X_val, y_val, metric=metric)

    # 3. Формирование результата с расчетом свойств эскалации
    return EvalResult(
        val_score=val_score,
        train_score=train_score,
        dummy_score=dummy_score,
        metric_name=metric.lower(),
        threshold=threshold,
    )
