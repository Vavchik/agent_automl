"""
Стадия 4: Стратегии разбиения данных (Time, Group, Stratified, K-Fold).

Чистый модуль машинного обучения без LLM-зависимостей:
- `detect_split_strategy`: автоматическое определение оптимальной стратегии сплита.
- `split_data`: разбиение DataFrame на train, test (и опционально val).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from ml_core.schemas import SplitResult


def detect_split_strategy(
    df: pd.DataFrame,
    target: str,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> str:
    """
    Автоматически определяет стратегию разбиения:
    - 'time_series': если указана временная колонка;
    - 'group': если есть группирующий признак (ID пользователя, сессия);
    - 'stratified': для задач классификации (сохранение баланса классов);
    - 'kfold': случайное разбиение для непрерывной регрессии.
    """
    if time_col and time_col in df.columns:
        return "time_series"
    if group_col and group_col in df.columns:
        return "group"

    target_s = df[target].dropna()
    is_float = pd.api.types.is_float_dtype(target_s)
    is_cat = isinstance(target_s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(target_s) or pd.api.types.is_object_dtype(target_s)
    if is_cat:
        return "stratified"
    if not is_float and target_s.nunique() <= 20:
        return "stratified"
    return "kfold"


def split_data(
    df: pd.DataFrame,
    target: str,
    strategy: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.0,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> SplitResult:
    """
    Разбивает датасет на обучающую, тестовую (и опционально валидационную) выборки.

    Args:
        df: исходный датасет.
        target: целевая переменная.
        strategy: 'stratified', 'time_series', 'group', 'kfold' (или None для автодетекта).
        test_size: доля тестовой выборки (например, 0.2).
        val_size: доля валидационной выборки от исходного датасета (например, 0.1).
        time_col: имя колонки времени для time_series.
        group_col: имя колонки групп для group.
        random_state: сид генератора случайных чисел.

    Returns:
        Объект SplitResult с train, test, val датафреймами.
    """
    if target not in df.columns:
        raise KeyError(f"Целевая колонка '{target}' не найдена в таблице")

    if strategy is None:
        strategy = detect_split_strategy(df, target, time_col=time_col, group_col=group_col)

    # 1. Временное разбиение (Time Series) — строго без заглядывания в будущее
    if strategy == "time_series":
        df_sorted = df.sort_values(by=time_col) if time_col else df
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size) if val_size > 0 else 0
        n_train = n_total - n_test - n_val

        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train : n_train + n_val].copy() if n_val > 0 else None
        test_df = df_sorted.iloc[n_train + n_val :].copy()

    # 2. Групповое разбиение (Group Split) — предотвращает утечки по группам
    elif strategy == "group":
        if not group_col or group_col not in df.columns:
            raise ValueError("Для стратегии 'group' необходимо указать group_col")

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        val_df = None

        if val_size > 0:
            adjusted_val = val_size / (1.0 - test_size)
            val_splitter = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=random_state)
            tr_idx, v_idx = next(val_splitter.split(train_df, groups=train_df[group_col]))
            val_df = train_df.iloc[v_idx].copy()
            train_df = train_df.iloc[tr_idx].copy()

    # 3. Стратифицированное разбиение (Stratified Split) — для классификации
    elif strategy == "stratified":
        target_s = df[target].dropna()
        can_stratify = not pd.api.types.is_float_dtype(target_s) and not (target_s.value_counts() < 2).any()
        stratify_col = df[target] if can_stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state,
        )
        val_df = None
        if val_size > 0:
            adjusted_val = val_size / (1.0 - test_size)
            train_stratify = (
                train_df[target]
                if (stratify_col is not None and not (train_df[target].value_counts() < 2).any())
                else None
            )
            train_df, val_df = train_test_split(
                train_df,
                test_size=adjusted_val,
                stratify=train_stratify,
                random_state=random_state,
            )
        train_df = train_df.copy()
        test_df = test_df.copy()
        if val_df is not None:
            val_df = val_df.copy()

    # 4. Случайное разбиение (K-Fold / Random Split) — для регрессии
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )
        val_df = None
        if val_size > 0:
            adjusted_val = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(
                train_df,
                test_size=adjusted_val,
                random_state=random_state,
            )
        train_df = train_df.copy()
        test_df = test_df.copy()
        if val_df is not None:
            val_df = val_df.copy()

    return SplitResult(
        train=train_df,
        test=test_df,
        val=val_df,
        strategy=strategy,
        target=target,
        time_col=time_col,
        group_col=group_col,
    )

