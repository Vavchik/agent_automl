"""Р3. Преобразования таргета для обучения.

Признаки готовит task/dtypes.py (общие категории для train и test),
здесь — только то, что зависит от модели и таргета.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_labels(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Метки -> коды 0..k-1 (классы отсортированы). Возвращает (коды, классы)."""
    codes, classes = pd.factorize(y, sort=True)
    return codes.astype(int), np.asarray(classes)


class TargetTransform:
    """None — без изменений; "log1p" — для неотрицательного таргета с длинным хвостом."""

    def __init__(self, kind: str | None, y: np.ndarray):
        self.kind = kind if (kind == "log1p" and np.nanmin(y) > -1) else None

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y) if self.kind == "log1p" else y

    def inverse(self, p: np.ndarray) -> np.ndarray:
        return np.expm1(p) if self.kind == "log1p" else p
