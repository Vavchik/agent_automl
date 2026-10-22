"""Р2. Запись submission.csv строго по формату sample_submission.

Формат определяется TaskSpec.output:
  value            — число
  label            — метка класса
  proba            — вероятность положительного класса (второй класс в classes)
  proba_per_class  — по колонке вероятностей на класс, имена колонок как в sample_submission
Перед записью проверяем: число строк, уникальность и набор id, NaN/inf. Запись атомарная.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.contracts import TaskSpec


class SubmissionError(ValueError):
    pass


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def build_submission(spec: TaskSpec, test_ids: pd.Series, pred, classes=None) -> pd.DataFrame:
    id_col, *value_cols = spec.submission_columns
    sub = pd.DataFrame({id_col: test_ids.to_numpy()})
    pred = np.asarray(pred)
    if spec.output == "proba_per_class":
        by_name = {str(c): i for i, c in enumerate(classes)}
        for col in value_cols:
            sub[col] = pred[:, by_name[str(col)]] if str(col) in by_name else 0.0
    else:
        sub[value_cols[0]] = pred
    return sub


def validate_submission(spec: TaskSpec, sub: pd.DataFrame) -> None:
    sample = pd.read_csv(spec.sample_sub_path)
    if list(sub.columns) != list(sample.columns):
        raise SubmissionError(f"Колонки {list(sub.columns)} != {list(sample.columns)}")
    if len(sub) != len(sample):
        raise SubmissionError(f"Строк {len(sub)}, ожидалось {len(sample)}")
    ids = _norm_id(sub[sub.columns[0]])
    if ids.duplicated().any():
        raise SubmissionError("Повторяющиеся id")
    if set(ids) != set(_norm_id(sample[sample.columns[0]])):
        raise SubmissionError("Набор id не совпадает с sample_submission")
    values = sub[sub.columns[1:]]
    if values.isna().any().any():
        raise SubmissionError("В ответе есть пропуски")
    numeric = values.select_dtypes("number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SubmissionError("В ответе есть бесконечности")


def write_submission(spec: TaskSpec, test_ids: pd.Series, pred, classes=None) -> Path:
    sub = build_submission(spec, test_ids, pred, classes)
    validate_submission(spec, sub)
    out = spec.task_dir / "submission.csv"
    tmp = out.with_suffix(".csv.tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out
