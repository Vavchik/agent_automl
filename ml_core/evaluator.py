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
import pandas as pd
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

from ml_core.schemas import EvalResult


CLASSIFICATION_METRICS = {"roc_auc", "macro_f1", "f1_macro", "accuracy", "logloss", "multiclass"}


def _is_classification_metric(metric: str) -> bool:
    return metric.lower() in CLASSIFICATION_METRICS


def compute_metric(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
    metric: str = "roc_auc",
) -> float:
    """
    Вычисляет значение указанной метрики качества.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metric_norm = metric.lower()

    if metric_norm in ("roc_auc", "auc"):
        # Если вероятности переданы
        if y_proba is not None:
            proba_arr = np.asarray(y_proba)
            # Проверка на число классов
            unique_classes = np.unique(y_true_arr)
            if len(unique_classes) < 2:
                return 0.5

            if len(unique_classes) == 2:
                # Бинарная классификация: берем вероятность положительного класса
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                    p = proba_arr[:, 1]
                else:
                    p = proba_arr.ravel()
                try:
                    return float(roc_auc_score(y_true_arr, p))
                except Exception:
                    pass
            else:
                # Мультикласс
                try:
                    return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
                except Exception:
                    pass

        # Fallback если нет вероятностей
        try:
            return float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            return float(accuracy_score(y_true_arr, y_pred_arr))

    if metric_norm in ("macro_f1", "f1_macro", "f1"):
        return float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    if metric_norm == "accuracy":
        return float(accuracy_score(y_true_arr, y_pred_arr))

    if metric_norm in ("logloss", "multiclass"):
        if y_proba is not None:
            try:
                return float(log_loss(y_true_arr, y_proba))
            except Exception:
                pass
        return float(log_loss(y_true_arr, y_pred_arr))

    if metric_norm in ("rmse", "root_mean_squared_error"):
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        return float(np.sqrt(mse))

    if metric_norm in ("mae", "mean_absolute_error"):
        return float(mean_absolute_error(y_true_arr, y_pred_arr))

    if metric_norm in ("r2", "r2_score"):
        return float(r2_score(y_true_arr, y_pred_arr))

    # Fallback на RMSE для регрессии или Accuracy для классификации
    try:
        return float(mean_squared_error(y_true_arr, y_pred_arr))
    except Exception:
        return float(accuracy_score(y_true_arr, y_pred_arr))


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
