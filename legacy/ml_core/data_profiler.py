"""
Стадия 1: Сбор метрик (пропуски, nunique, распределения, корреляция с таргетом).

Принимает уже приведённый к типам датасет и словарь типов из `type_inference.infer_types`
и собирает `DataProfileReport` (см. `schemas.py`).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml_core.schemas import DataProfileReport
from ml_core.type_inference import CATEGORY, DATETIME, NUMERIC, TEXT

_NUM_LIKE = {NUMERIC, DATETIME}


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def _py(value: Any) -> Any:
    """numpy/pandas скаляры -> JSON-совместимые python-типы."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value if isinstance(value, (int, str, bool)) else str(value)


# --------------------------------------------------------------------------- #
# Распределения
# --------------------------------------------------------------------------- #

def _numeric_distribution(col: pd.Series, bins: int = 10) -> dict[str, Any]:
    values = col.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "min": _py(values.min()),
        "max": _py(values.max()),
        "mean": _py(values.mean()),
        "std": _py(values.std()),
        "q25": _py(values.quantile(0.25)),
        "median": _py(values.median()),
        "q75": _py(values.quantile(0.75)),
        "histogram": {
            "bin_edges": [_py(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


def _datetime_distribution(col: pd.Series) -> dict[str, Any]:
    values = col.dropna()
    if values.empty:
        return {}
    return {
        "min": _py(values.min()),
        "max": _py(values.max()),
        "span_days": _py((values.max() - values.min()).days),
    }


def _category_distribution(col: pd.Series, top: int = 10) -> dict[str, Any]:
    shares = col.astype(object).value_counts(normalize=True, dropna=True).head(top)
    return {"top_shares": {str(k): _py(v) for k, v in shares.items()}}


def _text_distribution(col: pd.Series, top: int = 5) -> dict[str, Any]:
    values = col.dropna().astype(str)
    lengths = values.str.len()
    return {
        "len_mean": _py(lengths.mean()),
        "len_min": _py(lengths.min()),
        "len_max": _py(lengths.max()),
        "top_values": {str(k): int(v) for k, v in values.value_counts().head(top).items()},
    }


def _distribution(col: pd.Series, ctype: str) -> dict[str, Any]:
    if ctype == NUMERIC:
        return _numeric_distribution(col)
    if ctype == DATETIME:
        return _datetime_distribution(col)
    if ctype == CATEGORY:
        return _category_distribution(col)
    if ctype == TEXT:
        return _text_distribution(col)
    return {}


# --------------------------------------------------------------------------- #
# Корреляция с таргетом
# --------------------------------------------------------------------------- #

def _to_float(col: pd.Series, ctype: str) -> pd.Series:
    if ctype == DATETIME:
        return (col - col.min()).dt.total_seconds()
    return col.astype(float)


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> Optional[float]:
    """Корреляционное отношение eta in [0, 1]: связь категории с числом."""
    ss_total = ((num - num.mean()) ** 2).sum()
    if ss_total == 0:
        return None
    groups = num.groupby(cat.astype(object))
    ss_between = (groups.count() * (groups.mean() - num.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total))


def _cramers_v(x: pd.Series, y: pd.Series) -> Optional[float]:
    """V Крамера in [0, 1]: связь двух категориальных признаков."""
    table = pd.crosstab(x.astype(object), y.astype(object)).to_numpy(dtype=float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n_rows, n_cols = table.shape
    if n_rows < 2 or n_cols < 2:
        return None
    total = table.sum()
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / total
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (total * (min(n_rows, n_cols) - 1))))


def _correlation(
    x: pd.Series, x_type: str, y: pd.Series, y_type: str
) -> Optional[tuple[str, Optional[float]]]:
    """Подбирает меру связи под пару типов. None - если считать нечего/нельзя."""
    supported = _NUM_LIKE | {CATEGORY}
    if x_type not in supported or y_type not in supported:
        return None

    pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    if len(pair) < 3:
        return None
    xs, ys = pair["x"], pair["y"]

    x_num, y_num = x_type in _NUM_LIKE, y_type in _NUM_LIKE
    if x_num and y_num:
        xs, ys = _to_float(xs, x_type), _to_float(ys, y_type)
        if xs.nunique() < 2 or ys.nunique() < 2:
            return "pearson", None
        return "pearson", _py(xs.corr(ys))
    if x_num != y_num:
        cat, num, num_type = (ys, xs, x_type) if x_num else (xs, ys, y_type)
        return "correlation_ratio", _py(_correlation_ratio(cat, _to_float(num, num_type)))
    return "cramers_v", _py(_cramers_v(xs, ys))


def _target_correlation(
    df: pd.DataFrame, column_types: dict[str, str], target: str, id_col: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if col == target or (id_col and col == id_col) or col.lower() in ("id", "index", "uid"):
            continue
        out = _correlation(df[col], column_types[col], df[target], column_types[target])
        if out is None:
            continue
        method, value = out
        result[col] = {"method": method, "value": value}
    # Сначала самые сильные связи
    return dict(
        sorted(result.items(), key=lambda kv: -abs(kv[1]["value"]) if kv[1]["value"] is not None else 1)
    )


# --------------------------------------------------------------------------- #
# Сборка результата
# --------------------------------------------------------------------------- #

def profile_data(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target: Optional[str] = None,
    id_col: Optional[str] = None,
    head_rows: int = 5,
) -> DataProfileReport:
    """
    Собирает отчёт о датасете.

    Args:
        df: датасет, приведённый к типам (результат `infer_types`).
        column_types: {колонка: тип} (результат `infer_types`).
        target: имя таргета; если задано - считается корреляция признаков с ним.
        id_col: имя колонки ID (исключается из расчета корреляций).
        head_rows: сколько первых строк включить в отчёт.
    """
    if target is not None and target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    n_rows = max(len(df), 1)
    return DataProfileReport(
        n_rows=len(df),
        column_types=column_types,
        missing_share={c: float(df[c].isna().sum() / n_rows) for c in df.columns},
        nunique={c: int(df[c].nunique(dropna=True)) for c in df.columns},
        distributions={c: _distribution(df[c], column_types[c]) for c in df.columns},
        head=json.loads(df.head(head_rows).to_json(orient="records", date_format="iso")),
        target=target,
        target_correlation=_target_correlation(df, column_types, target, id_col=id_col) if target else {},
    )
