"""Р2. Типы колонок и подготовка признаков — одно решение сразу для train и test.

Главное правило: любые словари (категории, типы) строятся по train+test вместе,
поэтому одна и та же категория получает один и тот же код в обеих выборках.
Таргет здесь не используется, утечки нет.

Что улучшать (Р2):
- даты: больше компонент (неделя года, праздники), разница между датами;
- текст: TF-IDF / длина / количество слов;
- ID-подобные колонки (почти все значения уникальны) — сейчас удаляются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.contracts import Dataset, TaskSpec
from runtime.logger import log

NUMERIC_PARSE_SHARE = 0.95   # строковая колонка считается числом, если парсится >= 95% значений
TEXT_MIN_AVG_LEN = 40        # средняя длина строки, начиная с которой колонка — «текст»


def _try_numeric(col: pd.Series) -> pd.Series | None:
    cleaned = col.astype(str).str.replace(",", ".", regex=False).str.replace(r"\s", "", regex=True)
    parsed = pd.to_numeric(cleaned.where(col.notna()), errors="coerce")
    if col.notna().sum() and parsed.notna().sum() >= NUMERIC_PARSE_SHARE * col.notna().sum():
        return parsed
    return None


def _try_datetime(col: pd.Series) -> pd.Series | None:
    sample = col.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d{1,4}[-./]\d{1,2}[-./]\d{1,4}").mean() > 0.9:
        return None
    parsed = pd.to_datetime(col, errors="coerce")
    return parsed if parsed.notna().mean() > 0.9 * col.notna().mean() else None


def prepare_dataset(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec) -> Dataset:
    y = train[spec.target_col].reset_index(drop=True)
    drop = [spec.target_col, spec.id_col]
    X = train.drop(columns=[c for c in drop if c in train.columns]).reset_index(drop=True)
    X_test = test.drop(columns=[c for c in drop if c in test.columns]).reset_index(drop=True)
    X_test = X_test.reindex(columns=X.columns)          # одинаковый набор и порядок колонок
    test_ids = test[spec.id_col].reset_index(drop=True)

    full = pd.concat([X, X_test], ignore_index=True)
    n_train = len(X)
    cat_cols: list[str] = []

    for col in list(full.columns):
        series = full[col]
        if pd.api.types.is_bool_dtype(series):
            full[col] = series.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        numeric = _try_numeric(series)
        if numeric is not None:
            full[col] = numeric
            continue
        dates = _try_datetime(series)
        if dates is not None:
            full[f"{col}__year"] = dates.dt.year
            full[f"{col}__month"] = dates.dt.month
            full[f"{col}__dayofweek"] = dates.dt.dayofweek
            full[f"{col}__dayofyear"] = dates.dt.dayofyear
            full = full.drop(columns=[col])
            continue
        as_str = series.astype("string")
        if as_str.str.len().mean() >= TEXT_MIN_AVG_LEN:
            full[f"{col}__len"] = as_str.str.len()
            full[f"{col}__words"] = as_str.str.split().str.len()
            full = full.drop(columns=[col])
            continue
        if series.nunique() > 0.9 * len(series):
            full = full.drop(columns=[col])          # почти уникальные строки — это ID, а не признак
            continue
        full[col] = as_str.astype("category")        # общие категории для train и test
        cat_cols.append(col)

    constant = [c for c in full.columns if full[c].nunique(dropna=False) <= 1]
    full = full.drop(columns=constant)
    cat_cols = [c for c in cat_cols if c in full.columns]
    full = full.replace([np.inf, -np.inf], np.nan)

    if spec.task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
    keep = y.notna().to_numpy()
    X_out = full.iloc[:n_train][keep].reset_index(drop=True)
    y_out = y[keep].reset_index(drop=True)
    log.info(f"Признаков: {full.shape[1]} (категориальных {len(cat_cols)}), удалено константных: {len(constant)}")
    return Dataset(X=X_out, y=y_out, X_test=full.iloc[n_train:].reset_index(drop=True),
                   test_ids=test_ids, cat_cols=cat_cols)


def feature_schema(ds: Dataset, spec: TaskSpec, importance: dict[str, float] | None = None,
                   max_features: int = 25) -> dict:
    """Компактная схема данных для промптов LLM (~1-2 тыс. токенов): самые важные признаки,
    их тип, пропуски, диапазон или топ-значения, асимметрия и описание из словаря."""
    cols = list(ds.X.columns)
    if importance:
        total = sum(importance.values()) or 1.0
        cols = sorted(cols, key=lambda c: -importance.get(c, 0.0))
    features = []
    for col in cols[:max_features]:
        s = ds.X[col]
        item: dict = {"name": col, "missing": round(float(s.isna().mean()), 3)}
        if col in ds.cat_cols:
            item.update(type="category", unique=int(s.nunique()), top=[str(v)[:30] for v in s.value_counts().index[:3]])
        else:
            q = s.quantile([0.01, 0.5, 0.99])
            item.update(type="number", range_p1_p99=[float(f"{q.iloc[0]:.3g}"), float(f"{q.iloc[2]:.3g}")],
                        median=float(f"{q.iloc[1]:.3g}"))
            if s.nunique() <= 20:
                item["unique"] = int(s.nunique())
            skew = s.skew()
            if pd.notna(skew) and abs(skew) > 1:
                item["skew"] = round(float(skew), 1)
        if importance:
            item["importance"] = round(importance.get(col, 0.0) / total, 3)
        desc = spec.feature_descriptions.get(col)
        if desc:
            item["description"] = desc[:70]
        features.append(item)
    if spec.is_classification:
        shares = ds.y.value_counts(normalize=True)
        target = {"classes": {str(k): round(float(v), 4) for k, v in shares.items()}}
    else:
        y = ds.y
        target = {"min": float(f"{y.min():.4g}"), "median": float(f"{y.median():.4g}"),
                  "max": float(f"{y.max():.4g}"), "skew": round(float(y.skew()), 2)}
    return {
        "task": spec.task_type, "metric": spec.metric, "target": spec.target_col, "target_stats": target,
        "rows": {"train": len(ds.X), "test": len(ds.X_test)}, "n_features": ds.X.shape[1],
        "time_col": spec.time_col, "features": features,
        "other_features": cols[max_features:max_features + 40],
    }
