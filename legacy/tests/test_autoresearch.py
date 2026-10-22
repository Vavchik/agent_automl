"""
Тесты для автономного цикла AutoResearch (agent/autoresearch.py).
"""

from pathlib import Path
import numpy as np
import pytest

from agent.autoresearch import AutoResearchLoop
from agent.memory import ExperimentMemory
from ml_core.dataset_loader import DatasetContext
from ml_core.trainer import CrossValidatedTrainer


def test_autoresearch_baseline_and_loop(tmp_path: Path):
    pd = pytest.importorskip("pandas")

    # Создаем фиктивную задачу
    task_dir = tmp_path / "dummy_task"
    task_dir.mkdir()

    # sample_submission.csv
    sample_sub_path = task_dir / "sample_submission.csv"
    sample_sub_df = pd.DataFrame({"id": [101, 102, 103, 104], "target": [0.0, 0.0, 0.0, 0.0]})
    sample_sub_df.to_csv(sample_sub_path, index=False)

    ctx = DatasetContext(
        task_dir=task_dir,
        task_md_path=task_dir / "task.md",
        task_text="Тестовая задача предсказания энергопотребления",
        train_path=task_dir / "train.csv",
        test_path=task_dir / "test.csv",
        dict_path=None,
        sample_sub_path=sample_sub_path,
        target_col="target",
        id_col="id",
        metric="rmse",
        task_type="regression",
        known_types={"f1": "numeric", "f2": "numeric", "id": "id", "target": "target"},
        feature_descriptions={"f1": "Площадь", "f2": "Температура"},
    )

    # Синтетические данные
    n_train = 50
    df_train = pd.DataFrame({
        "id": list(range(n_train)),
        "f1": np.linspace(1, 10, n_train),
        "f2": np.linspace(10, 20, n_train),
        "target": np.linspace(10, 50, n_train) + np.random.normal(0, 0.5, n_train),
    })

    df_test = pd.DataFrame({
        "id": [101, 102, 103, 104],
        "f1": [2.5, 4.5, 6.5, 8.5],
        "f2": [12.0, 14.0, 16.0, 18.0],
    })

    trainer = CrossValidatedTrainer(task_type="regression", metric="rmse")
    loop = AutoResearchLoop(
        ctx=ctx,
        trainer=trainer,
        max_iterations=2,
        time_budget_sec=60.0,
    )

    # 1. Запуск бейзлайна
    base_score, sub_path = loop.run_baseline(df_train, df_test)
    assert base_score > 0.0
    assert sub_path.exists()
    assert sub_path.stat().st_size > 0

    # Проверяем структуру сабмита
    saved_sub = pd.read_csv(sub_path)
    assert list(saved_sub.columns) == ["id", "target"]
    assert len(saved_sub) == 4
    assert list(saved_sub["id"]) == [101, 102, 103, 104]

    # 2. Запуск цикла исследователя
    res = loop.run_research(df_train, df_test)
    assert res.best_score > 0.0
    assert res.memory is not None
    assert len(res.memory.records) >= 2  # baseline + at least 1 iteration


def test_autoresearch_pure_numpy(tmp_path: Path):
    import csv

    task_dir = tmp_path / "numpy_task"
    task_dir.mkdir()

    sample_sub_path = task_dir / "sample_submission.csv"
    with open(sample_sub_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target"])
        writer.writerow(["101", "0.0"])
        writer.writerow(["102", "0.0"])
        writer.writerow(["103", "0.0"])

    ctx = DatasetContext(
        task_dir=task_dir,
        task_md_path=task_dir / "task.md",
        task_text="Тестовая задача",
        train_path=task_dir / "train.csv",
        test_path=task_dir / "test.csv",
        dict_path=None,
        sample_sub_path=sample_sub_path,
        target_col="target",
        id_col="id",
        metric="rmse",
        task_type="regression",
        known_types={"id": "id", "target": "target"},
    )

    n_train = 30
    train_data = {
        "target": np.linspace(5, 25, n_train),
        "features": np.random.randn(n_train, 3),
    }
    test_data = {
        "ids": ["101", "102", "103"],
        "features": np.random.randn(3, 3),
    }

    trainer = CrossValidatedTrainer(task_type="regression", metric="rmse")
    loop = AutoResearchLoop(
        ctx=ctx,
        trainer=trainer,
        max_iterations=1,
        time_budget_sec=60.0,
    )

    score, sub_path = loop.run_baseline(train_data, test_data)
    assert score > 0.0
    assert sub_path.exists()

    with open(sub_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 4
    assert reader[0] == ["id", "target"]
    assert [r[0] for r in reader[1:]] == ["101", "102", "103"]
