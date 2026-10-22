"""Общие структуры данных между ролями.

Менять только через PR с аппрувом Р1 и предупреждением в чате:
от этих полей зависят все остальные папки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TaskSpec:
    """Что за задача и в каком виде сдавать ответ. Заполняет Р2 (task/)."""
    task_dir: Path
    text: str
    train_path: Path
    test_path: Path
    sample_sub_path: Path
    dict_path: Optional[Path]
    target_col: str
    id_col: str
    task_type: str                 # "regression" | "binary" | "multiclass"
    metric: str                    # ключ из task/metrics.py
    output: str                    # "value" | "label" | "proba" | "proba_per_class"
    submission_columns: list[str]
    time_col: Optional[str] = None     # колонка периода, если test лежит «в будущем» (Р5)
    group_col: Optional[str] = None    # колонка сущности для GroupKFold (Р5)
    feature_descriptions: dict[str, str] = field(default_factory=dict)
    notes: str = ""                # что агент понял из текста задачи

    @property
    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass")


@dataclass
class Dataset:
    """Признаки, готовые к обучению. Строит Р2 (task/dtypes.py), меняет Р4 (новые признаки)."""
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    test_ids: pd.Series
    cat_cols: list[str]


@dataclass
class ModelConfig:
    """Как обучать. Агент (Р4) меняет поля, ML-ядро (Р3) исполняет."""
    models: list[str] = field(default_factory=lambda: ["lgbm"])
    params: dict[str, dict] = field(default_factory=dict)   # переопределения параметров по моделям
    target_transform: Optional[str] = None                  # None | "log1p"

    def copy(self) -> "ModelConfig":
        return ModelConfig(list(self.models), {k: dict(v) for k, v in self.params.items()}, self.target_transform)


@dataclass
class ExperimentResult:
    """Результат одного эксперимента на фиксированных фолдах. Возвращает Р3, судит Р5."""
    name: str
    score: float                     # честный OOF-скор по метрике задачи
    fold_scores: list[float]
    oof: np.ndarray                  # регрессия: (n,), классификация: вероятности (n, k)
    test_pred: np.ndarray            # то же для test, усреднено по фолдам
    classes: Optional[np.ndarray]    # метки классов в порядке колонок вероятностей
    seconds: float
    feature_importance: dict[str, float] = field(default_factory=dict)


Folds = list[tuple[np.ndarray, np.ndarray]]
