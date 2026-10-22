"""
Модуль пост-процессинга и калибровки предсказаний под целевые метрики:
1. Macro F1: оптимизация весов классов для редких категорий (threshold / weight tuning).
2. RMSE: клиппинг недопустимых значений (неотрицательность) и экспоненциальное обратное преобразование.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ml_core.evaluator import compute_metric


def optimize_class_weights(
    oof_probs: np.ndarray,
    y_true: Sequence[Any],
    classes: Sequence[Any],
    metric: str = "macro_f1",
) -> np.ndarray:
    """
    Оптимизирует вектор весов w для классов argmax(w_k * p_{ik}) с целью максимизации Macro F1.
    Особенно критично для дисбалансированных задач (например, редкие сетевые атаки).
    """
    oof_probs = np.asarray(oof_probs, dtype=float)
    y_true_arr = np.asarray(y_true)
    classes_arr = np.asarray(classes)
    n_classes = len(classes_arr)

    if n_classes <= 2:
        return np.ones(n_classes, dtype=float)

    # Базовые предсказания без весов
    base_preds = classes_arr[np.argmax(oof_probs, axis=1)]
    base_score = compute_metric(y_true_arr, base_preds, metric=metric)

    best_weights = np.ones(n_classes, dtype=float)
    best_score = base_score

    # 1. Быстрый координатный перебор (Coordinate Sweep)
    candidate_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0]
    for c_idx in range(n_classes):
        current_best_mult = best_weights[c_idx]
        for mult in candidate_multipliers:
            trial_weights = best_weights.copy()
            trial_weights[c_idx] = mult
            weighted_probs = oof_probs * trial_weights
            preds = classes_arr[np.argmax(weighted_probs, axis=1)]
            score = compute_metric(y_true_arr, preds, metric=metric)
            if score > best_score + 1e-4:
                best_score = score
                current_best_mult = mult
        best_weights[c_idx] = current_best_mult

    # 2. Тонкая оптимизация через scipy (если доступна)
    if HAS_SCIPY and best_score > base_score:
        def loss_fn(weights: np.ndarray) -> float:
            w_pos = np.clip(weights, 0.1, 20.0)
            weighted_probs = oof_probs * w_pos
            preds = classes_arr[np.argmax(weighted_probs, axis=1)]
            score = compute_metric(y_true_arr, preds, metric=metric)
            return -score  # минимизируем отрицательный F1

        try:
            res = minimize(
                loss_fn,
                best_weights,
                method="Nelder-Mead",
                options={"maxiter": 100, "xatol": 0.05, "fatol": 0.001},
            )
            if -res.fun > best_score:
                best_weights = np.clip(res.x, 0.1, 20.0)
        except Exception:
            pass

    return best_weights


def apply_calibrated_predictions(
    probs: np.ndarray,
    classes: Sequence[Any],
    weights: Optional[np.ndarray] = None,
) -> list[Any]:
    """
    Применяет откалиброванные веса к вероятностям и возвращает метки классов.
    """
    probs_arr = np.asarray(probs, dtype=float)
    classes_arr = np.asarray(classes)

    if weights is not None:
        probs_arr = probs_arr * np.asarray(weights, dtype=float)

    best_indices = np.argmax(probs_arr, axis=1)
    return [classes_arr[idx] for idx in best_indices]


def postprocess_regression(
    preds: Sequence[float],
    lower_bound: Optional[float] = 0.0,
    upper_bound: Optional[float] = None,
    is_log1p: bool = False,
) -> list[float]:
    """
    Пост-процессинг для регрессии:
    1. Обратное преобразование expm1, если модель обучалась на log1p(y).
    2. Ограничение значений (клиппинг неотрицательности для энергопотребления).
    """
    preds_arr = np.asarray(preds, dtype=float)

    if is_log1p:
        preds_arr = np.expm1(preds_arr)

    a_min = lower_bound if lower_bound is not None else -np.inf
    a_max = upper_bound if upper_bound is not None else np.inf

    clipped = np.clip(preds_arr, a_min=a_min, a_max=a_max)
    return [float(x) for x in clipped]
