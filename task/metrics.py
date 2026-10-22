"""Р2. Реестр метрик: как считать, что лучше, и в каком виде нужен ответ.

needs:
  "value" — число (регрессия)
  "label" — метка класса
  "proba" — вероятности (binary: P(второй класс), multiclass: матрица n x k)
Чтобы добавить метрику: функция + строка в METRICS + шаблон в METRIC_PATTERNS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    fn: Callable
    higher_is_better: bool
    family: str          # "regression" | "classification"
    needs: str           # "value" | "label" | "proba"


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _rmsle(y, p):
    return float(np.sqrt(skm.mean_squared_log_error(np.clip(y, 0, None), np.clip(p, 0, None))))


def _roc_auc(y, p):
    if p.ndim == 2:
        return float(skm.roc_auc_score(y, p, multi_class="ovr", labels=np.arange(p.shape[1])))
    return float(skm.roc_auc_score(y, p))


def _logloss(y, p):
    labels = np.arange(p.shape[1]) if p.ndim == 2 else [0, 1]
    return float(skm.log_loss(y, p, labels=labels))


METRICS: dict[str, Metric] = {
    "rmse": Metric(_rmse, False, "regression", "value"),
    "mse": Metric(lambda y, p: float(skm.mean_squared_error(y, p)), False, "regression", "value"),
    "mae": Metric(lambda y, p: float(skm.mean_absolute_error(y, p)), False, "regression", "value"),
    "rmsle": Metric(_rmsle, False, "regression", "value"),
    "mape": Metric(lambda y, p: float(skm.mean_absolute_percentage_error(y, p)), False, "regression", "value"),
    "r2": Metric(lambda y, p: float(skm.r2_score(y, p)), True, "regression", "value"),
    "macro_f1": Metric(lambda y, p: float(skm.f1_score(y, p, average="macro")), True, "classification", "label"),
    "weighted_f1": Metric(lambda y, p: float(skm.f1_score(y, p, average="weighted")), True, "classification", "label"),
    "f1": Metric(lambda y, p: float(skm.f1_score(y, p)), True, "classification", "label"),
    "accuracy": Metric(lambda y, p: float(skm.accuracy_score(y, p)), True, "classification", "label"),
    "roc_auc": Metric(_roc_auc, True, "classification", "proba"),
    "pr_auc": Metric(lambda y, p: float(skm.average_precision_score(y, p)), True, "classification", "proba"),
    "logloss": Metric(_logloss, False, "classification", "proba"),
}

# Порядок важен только при одинаковой позиции в тексте (rmsle раньше rmse).
METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"macro[\s_-]?f1|f1[\s_-]?macro|макро[\s-]?f1", "macro_f1"),
    (r"weighted[\s_-]?f1|f1[\s_-]?weighted|взвешенн\w* f1", "weighted_f1"),
    (r"pr[\s_-]?auc|average[\s_]precision", "pr_auc"),
    (r"roc[\s_-]?auc|\bauc\b|gini|джини", "roc_auc"),
    (r"log[\s_-]?loss|кросс-энтропи|cross[\s-]?entropy", "logloss"),
    (r"rmsle", "rmsle"),
    (r"rmse|root mean squared", "rmse"),
    (r"\bmae\b|mean absolute error|средняя абсолютная", "mae"),
    (r"\bmape\b", "mape"),
    (r"\bmse\b", "mse"),
    (r"\br2\b|r²|r\^2|детерминации", "r2"),
    (r"\bf1\b", "f1"),
    (r"accuracy|доля правильных", "accuracy"),
]


def detect_metric(text: str) -> Optional[str]:
    """Ищет метрику, упомянутую первой после слова «метрика» (или во всём тексте)."""
    low = text.lower()
    start = low.find("метрик")
    if start < 0:
        start = low.find("metric")
    region = low[start:] if start >= 0 else low
    best: tuple[int, int, str] | None = None
    for order, (pattern, name) in enumerate(METRIC_PATTERNS):
        match = re.search(pattern, region)
        if match and (best is None or (match.start(), order) < best[:2]):
            best = (match.start(), order, name)
    return best[2] if best else None


def score(metric: str, y_true, pred) -> float:
    return METRICS[metric].fn(np.asarray(y_true), np.asarray(pred))


def improvement(metric: str, new: float, old: float) -> float:
    """Положительное число = new лучше old, с учётом направления метрики."""
    return (new - old) if METRICS[metric].higher_is_better else (old - new)
