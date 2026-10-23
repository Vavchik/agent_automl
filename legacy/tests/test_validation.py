"""Р5: схема фолдов и правило принятия."""
import numpy as np
import pandas as pd

from core.contracts import Dataset, ExperimentResult
from tests.test_task import _spec
from validation.acceptance import should_accept
from validation.splits import make_folds


def _ds(train_years, test_years):
    n = len(train_years)
    X = pd.DataFrame({"year": train_years, "x": np.arange(n)})
    X_test = pd.DataFrame({"year": test_years, "x": np.arange(len(test_years))})
    return Dataset(X, pd.Series(np.arange(n, dtype=float)), X_test, pd.Series(range(len(test_years))), [])


def test_time_split_when_test_is_later(tmp_path):
    spec = _spec(tmp_path)
    ds = _ds([1] * 10 + [2] * 10 + [3] * 10 + [4] * 10, [5] * 5)
    folds, scheme = make_folds(ds, spec, n_folds=4)
    assert spec.time_col == "year"
    for tr, va in folds:   # каждый фолд валидируется на одном целом годе
        assert ds.X["year"].iloc[va].nunique() == 1
        assert not set(ds.X["year"].iloc[tr]) & set(ds.X["year"].iloc[va])


def test_random_split_when_test_overlaps(tmp_path):
    spec = _spec(tmp_path)
    ds = _ds([1, 2, 3, 4] * 10, [2, 3])
    make_folds(ds, spec, n_folds=4)
    assert spec.time_col is None


def _result(fold_scores):
    return ExperimentResult("r", float(np.mean(fold_scores)), fold_scores, None, None, None, 0.0)


def test_accepts_stable_gain_and_rejects_noise():
    best = _result([10.0, 10.0, 10.0, 10.0])
    assert should_accept(_result([9.5, 9.6, 9.4, 9.5]), best, "rmse")[0]
    assert not should_accept(_result([8.0, 12.0, 8.5, 11.4]), best, "rmse")[0]
