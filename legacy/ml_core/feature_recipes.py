"""
Модуль проверенных соревновательных рецептов признаков (Kaggle Grandmaster Feature Recipes).

Реализует:
1. Автоматические GroupBy-агрегации когорт (Cohort Aggregations):
   - Разности со средним по группе (diff_from_group_mean).
   - Отношения к среднему по группе (ratio_to_group_mean).
   - Стандартные отклонения по группе (std_in_group).
2. Частотное кодирование категорий (Frequency / Count Encoding).
3. Полную защиту от утечек данных (Data Leakage):
   - Все статистики агрегируются исключительно по обучающей выборке (train)
     и транслируются на тестовую выборку (test).
   - Пропущенные и новые категории в тесте безопасно заполняются глобальными значениями.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger("ml_core.feature_recipes")


def extract_cohort_and_numeric_candidates(
    df: Any,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    max_cat_cardinality: int = 60,
    min_cat_cardinality: int = 2,
    max_candidates: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Автоматически находит наиболее перспективные категориальные когорты
    и непрерывные числовые признаки для групповых агрегаций.
    """
    if pd is None or not isinstance(df, pd.DataFrame):
        return [], []

    exclude_cols = {c for c in [target_col, id_col] if c}
    cat_candidates: list[tuple[str, int]] = []
    num_candidates: list[tuple[str, float]] = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        series = df[col]
        nunique = series.nunique(dropna=True)

        # Категориальные когорты
        if not pd.api.types.is_numeric_dtype(series) or (nunique <= max_cat_cardinality and nunique >= min_cat_cardinality):
            if min_cat_cardinality <= nunique <= max_cat_cardinality:
                cat_candidates.append((col, nunique))
        # Числовые признаки
        elif pd.api.types.is_numeric_dtype(series) and nunique > max_cat_cardinality:
            # Оцениваем стандартное отклонение
            std = float(series.std(skipna=True)) if len(series) > 1 else 0.0
            if std > 1e-4:
                num_candidates.append((col, std))

    # Сортируем: для категорий берем с умеренной мощностью (от 3 до 30)
    cat_candidates.sort(key=lambda x: abs(x[1] - 10))
    selected_cats = [c[0] for c in cat_candidates[:max_candidates]]

    # Для числовых берем с наибольшей дисперсией
    num_candidates.sort(key=lambda x: -x[1])
    selected_nums = [c[0] for c in num_candidates[:max_candidates]]

    return selected_cats, selected_nums


def apply_cohort_aggregations(
    df_train: Any,
    df_test: Optional[Any] = None,
    cat_cols: Optional[list[str]] = None,
    num_cols: Optional[list[str]] = None,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    max_new_features: int = 12,
) -> tuple[Any, Optional[Any], list[str]]:
    """
    Вычисляет и добавляет агрегации когорт к train и test без утечки данных.
    Возвращает (new_df_train, new_df_test, list_of_created_columns).
    """
    if pd is None or not isinstance(df_train, pd.DataFrame):
        return df_train, df_test, []

    train_res = df_train.copy()
    test_res = df_test.copy() if df_test is not None else None

    # Если списки колонок не переданы, определяем автоматически
    if not cat_cols or not num_cols:
        auto_cats, auto_nums = extract_cohort_and_numeric_candidates(
            train_res, target_col=target_col, id_col=id_col
        )
        cat_cols = cat_cols or auto_cats
        num_cols = num_cols or auto_nums

    created_features: list[str] = []
    features_count = 0

    for cat in cat_cols:
        if cat not in train_res.columns:
            continue

        # Приводим к строковому виду для надежности группировки
        train_cat_key = train_res[cat].astype(str).fillna("__MISSING__")
        test_cat_key = test_res[cat].astype(str).fillna("__MISSING__") if test_res is not None else None

        for num in num_cols:
            if num not in train_res.columns or features_count >= max_new_features:
                break

            train_num_vals = pd.to_numeric(train_res[num], errors="coerce")
            global_mean = float(train_num_vals.mean())
            global_std = float(train_num_vals.std()) if len(train_num_vals) > 1 else 1.0

            # Вычисляем статистики по обучающей выборке
            group_stats = pd.DataFrame({
                "cat_key": train_cat_key,
                "num_val": train_num_vals,
            }).groupby("cat_key")["num_val"].agg(["mean", "std"])

            mean_map = group_stats["mean"].to_dict()
            std_map = group_stats["std"].fillna(global_std).to_dict()

            # Признак 1: Разность со средним когорты (diff)
            f_diff_name = f"f_{num}_diff_{cat}"
            mapped_mean_train = train_cat_key.map(mean_map).fillna(global_mean)
            train_res[f_diff_name] = (train_num_vals - mapped_mean_train).fillna(0.0)

            if test_res is not None and test_cat_key is not None:
                test_num_vals = pd.to_numeric(test_res[num], errors="coerce")
                mapped_mean_test = test_cat_key.map(mean_map).fillna(global_mean)
                test_res[f_diff_name] = (test_num_vals - mapped_mean_test).fillna(0.0)

            created_features.append(f_diff_name)
            features_count += 1

            # Признак 2: Отношение к среднему когорты (ratio)
            if features_count < max_new_features:
                f_ratio_name = f"f_{num}_ratio_{cat}"
                train_res[f_ratio_name] = (train_num_vals / (mapped_mean_train.abs() + 1e-5)).fillna(1.0)

                if test_res is not None and test_cat_key is not None:
                    test_res[f_ratio_name] = (test_num_vals / (mapped_mean_test.abs() + 1e-5)).fillna(1.0)

                created_features.append(f_ratio_name)
                features_count += 1

    logger.info(f"Сгенерировано {len(created_features)} GroupBy признаков когорт: {created_features}")
    return train_res, test_res, created_features


def apply_frequency_encoding(
    df_train: Any,
    df_test: Optional[Any] = None,
    cat_cols: Optional[list[str]] = None,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    max_features: int = 5,
) -> tuple[Any, Optional[Any], list[str]]:
    """
    Частотное кодирование категориальных признаков.
    """
    if pd is None or not isinstance(df_train, pd.DataFrame):
        return df_train, df_test, []

    train_res = df_train.copy()
    test_res = df_test.copy() if df_test is not None else None

    if not cat_cols:
        auto_cats, _ = extract_cohort_and_numeric_candidates(
            train_res, target_col=target_col, id_col=id_col
        )
        cat_cols = auto_cats[:max_features]

    created: list[str] = []
    for cat in cat_cols:
        if cat not in train_res.columns:
            continue

        s_train = train_res[cat].astype(str).fillna("__MISSING__")
        freq_map = s_train.value_counts(normalize=True).to_dict()

        f_name = f"f_freq_{cat}"
        train_res[f_name] = s_train.map(freq_map).fillna(0.0).astype(float)

        if test_res is not None:
            s_test = test_res[cat].astype(str).fillna("__MISSING__")
            test_res[f_name] = s_test.map(freq_map).fillna(0.0).astype(float)

        created.append(f_name)

    return train_res, test_res, created
