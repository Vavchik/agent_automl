"""Р2: метрики, типы, сабмит."""
import numpy as np
import pandas as pd
import pytest

from core.contracts import TaskSpec
from task.dtypes import prepare_dataset
from task.metrics import detect_metric, improvement
from task.submission import SubmissionError, build_submission, validate_submission


@pytest.mark.parametrize("text, expected", [
    ("Метрика оценки — **RMSE** на исходной шкале", "rmse"),
    ("Поэтому основной метрикой выбран Macro F1, а не Accuracy. Дополнительно Weighted F1", "macro_f1"),
    ("Качество оценивается по ROC-AUC", "roc_auc"),
    ("Метрика — MAE, меньше лучше", "mae"),
    ("Метрика: RMSLE", "rmsle"),
    ("Метрика — LogLoss", "logloss"),
    ("Без упоминания", None),
])
def test_detect_metric(text, expected):
    assert detect_metric(text) == expected


def test_improvement_direction():
    assert improvement("rmse", 1.0, 2.0) > 0
    assert improvement("macro_f1", 0.6, 0.5) > 0


def _spec(tmp_path, **kw):
    base = dict(task_dir=tmp_path, text="", train_path=tmp_path, test_path=tmp_path,
                sample_sub_path=tmp_path / "sample_submission.csv", dict_path=None, target_col="y", id_col="id",
                task_type="regression", metric="rmse", output="value", submission_columns=["id", "y"])
    base.update(kw)
    return TaskSpec(**base)


def test_categories_shared_between_train_and_test(tmp_path):
    train = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "c", "a"], "x": [1, 2, 3, 4], "y": [1., 2, 3, 4]})
    test = pd.DataFrame({"id": [5, 6], "cat": ["c", "d"], "x": [5, 6]})
    ds = prepare_dataset(train, test, _spec(tmp_path))
    assert "cat" in ds.cat_cols
    assert list(ds.X["cat"].cat.categories) == list(ds.X_test["cat"].cat.categories)
    # одна и та же категория -> один и тот же код
    code_c_train = ds.X["cat"].cat.codes[2]
    code_c_test = ds.X_test["cat"].cat.codes[0]
    assert code_c_train == code_c_test


def test_dates_are_expanded(tmp_path):
    train = pd.DataFrame({"id": range(4), "d": ["2024-01-05", "2024-02-10", "2024-03-01", "2024-04-02"],
                          "y": [1., 2, 3, 4]})
    test = pd.DataFrame({"id": [9], "d": ["2024-05-06"]})
    ds = prepare_dataset(train, test, _spec(tmp_path))
    assert "d__month" in ds.X.columns and "d" not in ds.X.columns


def test_submission_proba_per_class(tmp_path):
    pd.DataFrame({"id": [1, 2], "A": [0.5, 0.5], "B": [0.5, 0.5]}).to_csv(tmp_path / "sample_submission.csv",
                                                                         index=False)
    spec = _spec(tmp_path, task_type="multiclass", metric="logloss", output="proba_per_class",
                 submission_columns=["id", "A", "B"])
    sub = build_submission(spec, pd.Series([1, 2]), np.array([[0.9, 0.1], [0.2, 0.8]]), np.array(["A", "B"]))
    validate_submission(spec, sub)
    assert sub["B"].tolist() == [0.1, 0.8]


def test_submission_rejects_nan(tmp_path):
    pd.DataFrame({"id": [1, 2], "y": [0, 0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec = _spec(tmp_path)
    with pytest.raises(SubmissionError):
        validate_submission(spec, build_submission(spec, pd.Series([1, 2]), [1.0, np.nan]))
