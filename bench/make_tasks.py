"""Р5 + Р2. Синтетические «private»-задачи в формате кейса, чтобы проверять универсальность.

python bench/make_tasks.py  ->  bench/tasks/<имя>/{task.md, train.csv, test.csv, ...}
Правильные ответы кладутся отдельно в bench/answers/<имя>.csv (агент их не видит).

Добавляйте сюда задачи, которых боитесь на private: другие метрики, даты, текст, строковые id,
отрицательный таргет, много классов, пропуски.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

ROOT = Path(__file__).resolve().parent


def _save(name: str, df: pd.DataFrame, target: str, id_col: str, task_text: str,
          sample_values: dict, descriptions: dict[str, str], out_root: Path = ROOT) -> Path:
    task_dir = out_root / "tasks" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "answers").mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    is_test = rng.rand(len(df)) < 0.3
    train, test = df[~is_test], df[is_test]
    train.to_csv(task_dir / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(task_dir / "test.csv", index=False)
    test[[id_col, target]].to_csv(out_root / "answers" / f"{name}.csv", index=False)
    sample = pd.DataFrame({id_col: test[id_col].to_numpy()})
    for col, value in sample_values.items():
        sample[col] = value
    sample.to_csv(task_dir / "sample_submission.csv", index=False)
    roles = {c: ("id" if c == id_col else "target" if c == target else "feature") for c in df.columns}
    pd.DataFrame({"feature": list(df.columns), "role": [roles[c] for c in df.columns],
                  "dtype": [str(df[c].dtype) for c in df.columns], "unit": "",
                  "description_ru": [descriptions.get(c, "") for c in df.columns]}
                 ).to_csv(task_dir / "data_dictionary.csv", index=False)
    (task_dir / "task.md").write_text(task_text, encoding="utf-8")
    return task_dir / "task.md"


def binary_auc(n: int = 6000, out_root: Path = ROOT) -> Path:
    X, y = make_classification(n, n_features=12, n_informative=6, weights=[0.85], random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["region"] = np.where(X[:, 0] > 0, "north", "south")
    df.loc[df.sample(frac=0.1, random_state=2).index, "f3"] = np.nan
    df.insert(0, "client_id", [f"c_{i:06d}" for i in range(n)])
    df["churn"] = y
    text = ("# Отток клиентов\n\nПо признакам клиента предскажите вероятность ухода (`churn` = 1).\n\n"
            "Метрика — **ROC-AUC**, в ответе вероятность класса 1.\n\n"
            "Формат: `submission.csv` с колонками `client_id,churn`.\n")
    return _save("binary_auc", df, "churn", "client_id", text, {"churn": 0.5}, {"region": "Регион"}, out_root)


def regression_mae_negative(n: int = 5000, out_root: Path = ROOT) -> Path:
    X, y = make_regression(n, n_features=8, n_informative=5, noise=10, random_state=3)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.random.RandomState(4).randint(0, 700, n), unit="D")
    df["event_date"] = dates.strftime("%Y-%m-%d")
    df.insert(0, "id", np.arange(n))
    df["profit"] = y - 150 + 30 * (dates.month.to_numpy() > 6)
    text = ("# Прибыль магазина\n\nПредскажите прибыль `profit` (может быть отрицательной).\n\n"
            "Метрика оценки — **MAE**, меньше — лучше.\n")
    return _save("regression_mae_negative", df, "profit", "id", text, {"profit": 0.0},
                 {"event_date": "Дата события"}, out_root)


def multiclass_logloss(n: int = 6000, out_root: Path = ROOT) -> Path:
    X, y = make_classification(n, n_features=10, n_informative=6, n_classes=3, random_state=5)
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(10)])
    df.insert(0, "id", np.arange(n))
    df["grade"] = np.array(["A", "B", "C"])[y]
    text = ("# Класс качества\n\nПредскажите вероятности классов `grade` (A, B, C).\n\n"
            "Метрика — **LogLoss**, в ответе по колонке вероятности на каждый класс.\n")
    return _save("multiclass_logloss", df, "grade", "id", text, {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, {}, out_root)


ALL = [binary_auc, regression_mae_negative, multiclass_logloss]

if __name__ == "__main__":
    for make in ALL:
        print(make())
