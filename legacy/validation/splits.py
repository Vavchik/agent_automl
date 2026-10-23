"""Р5. Схема валидации: фолды должны быть похожи на то, как устроен test.

Правила сейчас:
- есть колонка периода, и весь test лежит не раньше последнего периода train
  (например Year_Factor: train 1-5, test 6) -> GroupKFold по периодам:
  модель проверяется на годе, которого не видела;
- классификация -> StratifiedKFold;
- иначе -> KFold.
Фолды создаются один раз и используются всеми моделями — иначе скоры несравнимы.

Что улучшать (Р5):
- expanding window (учиться только на прошлом) вместо leave-one-period-out;
- GroupKFold по сущности (spec.group_col, сейчас только логируется);
- проверить, совпадает ли динамика CV с public score.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

import config
from core.contracts import Dataset, Folds, TaskSpec
from runtime.logger import log

MAX_PERIODS = 50   # колонка с большим числом значений — не «период»


def _is_later_period(train_col: pd.Series, test_col: pd.Series) -> bool:
    if not (pd.api.types.is_numeric_dtype(train_col) and pd.api.types.is_numeric_dtype(test_col)):
        return False
    n_periods = train_col.nunique()
    if not (3 <= n_periods <= MAX_PERIODS) or test_col.isna().any() or train_col.isna().any():
        return False
    return test_col.min() >= train_col.max()


def detect_time_col(ds: Dataset, spec: TaskSpec) -> Optional[str]:
    candidates = ([spec.time_col] if spec.time_col else []) + list(ds.X.columns)
    for col in candidates:
        if col in ds.X.columns and _is_later_period(ds.X[col], ds.X_test[col]):
            return col
    return None


def make_folds(ds: Dataset, spec: TaskSpec, n_folds: int = config.N_FOLDS) -> tuple[Folds, str]:
    idx = np.arange(len(ds.X))
    time_col = detect_time_col(ds, spec)
    if time_col:
        spec.time_col = time_col
        groups = ds.X[time_col].to_numpy()
        n_splits = min(n_folds, len(np.unique(groups)))
        folds = list(GroupKFold(n_splits=n_splits).split(idx, groups=groups))
        scheme = f"GroupKFold по периоду '{time_col}' ({n_splits} фолдов): test лежит в будущем"
    elif spec.is_classification:
        min_class = int(ds.y.value_counts().min())
        n_splits = max(2, min(n_folds, min_class))
        folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=config.SEED).split(idx, ds.y))
        scheme = f"StratifiedKFold ({n_splits} фолдов)"
    else:
        folds = list(KFold(n_folds, shuffle=True, random_state=config.SEED).split(idx))
        scheme = f"KFold ({n_folds} фолдов)"
    log.info(f"Валидация: {scheme}")
    if spec.group_col:
        log.info(f"LLM предлагает группировку по '{spec.group_col}' — пока не используется")
    return folds, scheme
