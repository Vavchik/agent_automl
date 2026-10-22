"""Р3. Перевод предсказаний модели в то, что нужно метрике и сабмиту.

- регрессия: обрезка по диапазону таргета в train (не по нулю — таргет может быть отрицательным);
- метрики по меткам (Macro F1 и т. п.): веса классов к вероятностям. Чтобы не обмануть себя,
  польза весов проверяется честно: подбираем на одной половине OOF, проверяем на другой.
  Если прироста нет — веса не применяем;
- метрики по вероятностям: binary -> P(второй класс), multiclass -> матрица.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

import config
from core.contracts import ExperimentResult, TaskSpec
from runtime.logger import log
from task.metrics import METRICS, score

MULTIPLIERS = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]


def to_metric_input(spec: TaskSpec, pred: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if spec.task_type == "regression":
        return pred
    if METRICS[spec.metric].needs == "label":
        proba = pred * weights if weights is not None else pred
        return proba.argmax(axis=1)
    return pred[:, 1] if spec.task_type == "binary" else pred


def evaluate(spec: TaskSpec, y_true: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """y_true: значения (регрессия) или коды классов 0..k-1."""
    return score(spec.metric, y_true, to_metric_input(spec, pred, weights))


def fit_class_weights(proba: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    """Покоординатный подбор множителей вероятностей классов под метрику по меткам."""
    weights = np.ones(proba.shape[1])
    best = score(metric, y, proba.argmax(axis=1))
    for _ in range(2):
        for k in range(proba.shape[1]):
            for m in MULTIPLIERS:
                trial = weights.copy()
                trial[k] = m
                s = score(metric, y, (proba * trial).argmax(axis=1))
                if s > best + 1e-6:
                    best, weights = s, trial
    return weights


def honest_class_weights(proba: np.ndarray, y: np.ndarray, metric: str) -> tuple[np.ndarray | None, float]:
    """Подбор на половине A, проверка на половине B и наоборот. Возвращает (веса или None, честный прирост)."""
    gains = []
    min_class = np.bincount(y).min()
    if min_class < 2:
        return None, 0.0
    for a, b in StratifiedKFold(2, shuffle=True, random_state=config.SEED).split(proba, y):
        w = fit_class_weights(proba[a], y[a], metric)
        gains.append(score(metric, y[b], (proba[b] * w).argmax(axis=1)) - score(metric, y[b], proba[b].argmax(axis=1)))
    gain = float(np.mean(gains))
    if gain <= 0:
        return None, gain
    return fit_class_weights(proba, y, metric), gain


def final_prediction(spec: TaskSpec, result: ExperimentResult, y_train: np.ndarray):
    """Предсказания test в виде для сабмита и классы (для proba_per_class)."""
    if spec.task_type == "regression":
        return np.clip(result.test_pred, np.min(y_train), np.max(y_train)), None
    needs = METRICS[spec.metric].needs
    if needs == "label":
        weights, gain = honest_class_weights(result.oof, np.asarray(y_train), spec.metric)
        if weights is not None:
            log.info(f"Веса классов дают честный прирост {gain:+.5f}: {np.round(weights, 2).tolist()}")
        codes = to_metric_input(spec, result.test_pred, weights)
        return result.classes[codes], result.classes
    if spec.output == "proba":
        return result.test_pred[:, 1], result.classes
    return result.test_pred, result.classes
