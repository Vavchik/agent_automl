"""Р2. Типы колонок и подготовка признаков — одно решение сразу для train и test.

Главное правило: любые словари (категории, типы) строятся по train+test вместе,
поэтому одна и та же категория получает один и тот же код в обеих выборках.
Таргет здесь не используется, утечки нет.

Что улучшать (Р2):
- даты: больше компонент (неделя года, праздники), разница между датами;
- текст: TF-IDF / длина / количество слов;
- ID-подобные колонки (почти все значения уникальны) — сейчас удаляются.
"""
import json
from typing import Any

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


def profile_text(ds: Dataset, spec: TaskSpec, max_cols: int = 80) -> str:
    """Короткий профиль данных для промптов LLM."""
    lines = []
    for col in list(ds.X.columns)[:max_cols]:
        s = ds.X[col]
        miss = f"{s.isna().mean():.0%} пропусков"
        if col in ds.cat_cols:
            info = f"категория, {s.nunique()} значений, топ: {list(s.value_counts().index[:3])}"
        else:
            info = f"число, min {s.min():.4g}, median {s.median():.4g}, max {s.max():.4g}"
        desc = spec.feature_descriptions.get(col, "")
        lines.append(f"- {col}: {info}, {miss}. {desc}".strip())
    if spec.is_classification:
        target = f"классы: {ds.y.value_counts().to_dict()}"
    else:
        target = f"min {ds.y.min():.4g}, median {ds.y.median():.4g}, max {ds.y.max():.4g}, skew {ds.y.skew():.2f}"
    return f"Таргет {spec.target_col}: {target}\nПризнаки:\n" + "\n".join(lines)


def build_schema_json(ds: Dataset, spec: TaskSpec, importance: dict[str, float] | None = None,
                      shifted: list[str] | None = None, max_cols: int = 20) -> str:
    """Генерирует компактную (<1500 токенов) JSON-схему задачи для Schema-First промптинга Qwen."""
    schema: dict[str, Any] = {
        "task": spec.task_type,
        "metric": spec.metric,
        "target": spec.target_col,
    }
    if spec.is_classification:
        counts = ds.y.value_counts(normalize=True).round(4).to_dict()
        schema["classes_distribution"] = {str(k): float(v) for k, v in counts.items()}
    else:
        y_num = pd.to_numeric(ds.y, errors="coerce")
        schema["target_stats"] = {
            "min": round(float(y_num.min()), 4),
            "median": round(float(y_num.median()), 4),
            "max": round(float(y_num.max()), 4),
            "skew": round(float(y_num.skew()), 2),
        }

    cols = list(ds.X.columns)
    if importance:
        cols.sort(key=lambda c: importance.get(c, 0.0), reverse=True)
    selected_cols = cols[:max_cols]
    total_imp = sum(importance.values()) if importance else 0.0

    features_meta = []
    sample_values = {}
    for col in selected_cols:
        s = ds.X[col]
        meta: dict[str, Any] = {"name": col, "dtype": str(s.dtype)}
        if importance and col in importance:
            meta["importance"] = round(float(importance[col]) / max(total_imp, 1e-12), 4)
        miss = float(s.isna().mean())
        if miss > 0:
            meta["missing_share"] = round(miss, 3)
        if col in ds.cat_cols:
            meta["cardinality"] = int(s.nunique())
            samples = s.dropna().astype(str).unique()[:3].tolist()
        else:
            meta["cardinality"] = int(s.nunique())
            s_num = pd.to_numeric(s, errors="coerce")
            if len(s_num.dropna()) > 0:
                meta["skew"] = round(float(s_num.skew()), 2)
            samples = [round(float(v), 3) if isinstance(v, (int, float, np.number)) else v
                       for v in s.dropna().unique()[:3]]
        features_meta.append(meta)
        sample_values[col] = samples[:3]

    schema["top_features"] = features_meta
    schema["sample_values"] = sample_values
    if shifted:
        schema["shifted_features"] = shifted[:10]
    if spec.notes:
        schema["task_notes"] = spec.notes[:300]
    return json.dumps(schema, ensure_ascii=False, indent=2)
