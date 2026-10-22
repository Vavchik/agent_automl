"""
Стадия 4: Стратегии разбиения данных (Time, Group, Stratified, K-Fold).

Чистый модуль машинного обучения без LLM-зависимостей:
- `detect_split_strategy`: автоматическое определение оптимальной стратегии сплита.
- `split_data`: разбиение DataFrame на train, test (и опционально val).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        train_test_split,
    )
except ImportError:
    GroupKFold = GroupShuffleSplit = KFold = StratifiedKFold = TimeSeriesSplit = train_test_split = None

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


def get_cv_folds(
    df: pd.DataFrame,
    target: str,
    strategy: Optional[str] = None,
    n_splits: int = 5,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> list[tuple[Any, Any]]:
    """
    Генерирует список кортежей (train_idx, val_idx) для честной K-Fold кросс-валидации.
    """
    if strategy is None:
        strategy = detect_split_strategy(df, target, time_col=time_col, group_col=group_col)

    if strategy == "time_series":
        df_sorted = df.sort_values(by=time_col) if time_col else df
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(df_sorted))

    if strategy == "group" and group_col and group_col in df.columns:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(df, groups=df[group_col]))

    if strategy == "stratified" and StratifiedKFold is not None and pd is not None and isinstance(df, pd.DataFrame):
        target_s = df[target].dropna()
        can_stratify = not pd.api.types.is_float_dtype(target_s) and not (target_s.value_counts() < n_splits).any()
        if can_stratify:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(skf.split(df, df[target]))

    if KFold is not None and pd is not None and isinstance(df, pd.DataFrame):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(df))

    # Чистый NumPy fallback (когда scikit-learn отсутствует или передан ndarray)
    n_samples = len(df)
    indices = np.arange(n_samples)
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    folds = []
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        tr_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((tr_idx, val_idx))
        current = stop
    return folds


def split_data(
    df: pd.DataFrame,
    target: str,
    strategy: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.0,
    n_folds: int = 5,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> SplitResult:
    """
    Разбивает датасет на обучающую, тестовую (и опционально валидационную) выборки
    и генерирует K-Fold сплиты на обучающей части.
    """
    if target not in df.columns:
        raise KeyError(f"Целевая колонка '{target}' не найдена в таблице")

    if strategy is None:
        strategy = detect_split_strategy(df, target, time_col=time_col, group_col=group_col)

    if test_size <= 0.0:
        # Без отложенного теста (например, когда test.csv уже дан отдельно организаторами)
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)
        val_df = None
    elif strategy == "time_series":
        df_sorted = df.sort_values(by=time_col) if time_col else df
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size) if val_size > 0 else 0
        n_train = n_total - n_test - n_val

        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train : n_train + n_val].copy() if n_val > 0 else None
        test_df = df_sorted.iloc[n_train + n_val :].copy()
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

    # Генерация K-Fold сплитов на train_df
    folds = None
    if n_folds > 1 and len(train_df) >= n_folds:
        folds = get_cv_folds(
            df=train_df,
            target=target,
            strategy=strategy,
            n_splits=n_folds,
            time_col=time_col,
            group_col=group_col,
            random_state=random_state,
        )

    return SplitResult(
        train=train_df,
        test=test_df,
        val=val_df,
        strategy=strategy,
        target=target,
        time_col=time_col,
        group_col=group_col,
        folds=folds,
    )

