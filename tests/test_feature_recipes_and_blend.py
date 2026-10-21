"""
Тесты для модуля Grandmaster Feature Recipes (ml_core/feature_recipes.py)
и режима Dual-Model Blending (ml_core/trainer.py).
"""

import numpy as np
import pandas as pd
import pytest

from ml_core.feature_recipes import (
    apply_cohort_aggregations,
    apply_frequency_encoding,
    extract_cohort_and_numeric_candidates,
)
from ml_core.trainer import CrossValidatedTrainer
from agent.feature_engineer import execute_feature_transform


def test_extract_cohort_and_numeric_candidates():
    df = pd.DataFrame({
        "id": range(100),
        "target": np.random.randn(100),
        "cat_col": ["A", "B", "C", "D"] * 25,
        "num_col1": np.random.randn(100) * 10,
        "num_col2": np.random.randn(100) * 5,
        "high_card": [f"val_{i}" for i in range(100)],
    })

    cats, nums = extract_cohort_and_numeric_candidates(
        df, target_col="target", id_col="id", max_cat_cardinality=10
    )
    assert "cat_col" in cats
    assert "num_col1" in nums
    assert "target" not in cats and "target" not in nums
    assert "id" not in cats and "id" not in nums


def test_apply_cohort_aggregations_and_unseen_categories():
    df_train = pd.DataFrame({
        "cat": ["A", "A", "B", "B", "B"],
        "val": [10.0, 20.0, 100.0, 200.0, 300.0],
        "target": [1, 2, 3, 4, 5],
    })
    df_test = pd.DataFrame({
        "cat": ["A", "B", "UNSEEN"],  # UNSEEN category in test
        "val": [15.0, 150.0, 50.0],
    })

    new_tr, new_te, created = apply_cohort_aggregations(
        df_train=df_train,
        df_test=df_test,
        cat_cols=["cat"],
        num_cols=["val"],
        target_col="target",
    )

    assert len(created) == 2  # diff and ratio
    assert "f_val_diff_cat" in new_tr.columns
    assert "f_val_ratio_cat" in new_tr.columns

    # Mean for A in train is 15.0; Mean for B is 200.0
    # For A in train: 10 - 15 = -5; 20 - 15 = 5
    assert np.isclose(new_tr["f_val_diff_cat"].iloc[0], -5.0)

    # Test set check (unseen category falls back safely to global mean, no NaNs/Infs)
    assert not new_te.isna().any().any()
    num_sub = new_te[["val", "f_val_diff_cat", "f_val_ratio_cat"]]
    assert not np.isinf(num_sub.to_numpy(dtype=float)).any()


def test_apply_frequency_encoding():
    df_train = pd.DataFrame({"cat": ["A", "A", "A", "B"]})
    df_test = pd.DataFrame({"cat": ["A", "B", "C"]})

    new_tr, new_te, created = apply_frequency_encoding(
        df_train=df_train,
        df_test=df_test,
        cat_cols=["cat"],
    )
    assert "f_freq_cat" in created
    assert np.isclose(new_tr["f_freq_cat"].iloc[0], 0.75)
    assert np.isclose(new_tr["f_freq_cat"].iloc[3], 0.25)
    assert not new_te["f_freq_cat"].isna().any()


def test_dual_model_blend_regression():
    np.random.seed(42)
    n = 120
    X = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "cat": np.random.choice(["type_1", "type_2", "type_3"], size=n),
    })
    y = X["x1"] * 2.5 + X["x2"] * -1.2 + np.random.randn(n) * 0.1

    trainer = CrossValidatedTrainer(
        task_type="regression",
        metric="rmse",
        algorithm="Blend",
        random_state=42,
    )

    res = trainer.train_cv(X_train=X, y_train=y, X_test=X.iloc[:20], n_estimators=30)
    assert res.oof_score > 0
    assert len(res.oof_preds) == n
    assert len(res.test_preds) == 20
    assert len(res.models) > 0


def test_execute_feature_transform_target_isolation_and_categorical_safety():
    df_train = pd.DataFrame({
        "id": [1, 2, 3],
        "target": [10.0, 20.0, 30.0],
        "cat": pd.Categorical(["low", "med", "high"]),
        "num": [1.0, 2.0, 3.0],
    })
    df_test = pd.DataFrame({
        "id": [4, 5],
        "cat": pd.Categorical(["low", "high"]),
        "num": [4.0, 5.0],
    })

    # Code that would normally fail on Categorical setitem or leak target:
    # 1. Calls .fillna(0) on categorical
    # 2. Attempts to create ratio
    risky_code = """def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Testing categorical modification:
    df['cat'] = df['cat'].fillna('missing')
    df['f_double_num'] = df['num'] * 2.0
    return df
"""
    new_tr, new_te, err = execute_feature_transform(
        code_str=risky_code,
        df_train=df_train,
        df_test=df_test,
        target_col="target",
    )
    assert err is None
    assert "f_double_num" in new_tr.columns
    assert "f_double_num" in new_te.columns
    # Target is preserved in train, absent in test
    assert "target" in new_tr.columns
    assert "target" not in new_te.columns
