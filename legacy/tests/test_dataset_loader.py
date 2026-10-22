"""
Тесты для модуля интеллектуальной загрузки датасетов (ml_core.dataset_loader).
"""

from pathlib import Path
from ml_core.dataset_loader import load_dataset_context

GIVEN_DATA_DIR = Path(__file__).resolve().parent.parent / "given_data"


def test_load_multiclass_task_context():
    task_path = GIVEN_DATA_DIR / "Многоклассовая классификация" / "task.md"
    assert task_path.exists(), f"Файл не найден: {task_path}"

    ctx = load_dataset_context(task_path)
    assert ctx.target_col == "attack_cat"
    assert ctx.id_col == "id"
    assert ctx.task_type == "multiclass_classification"
    assert ctx.metric == "macro_f1"
    assert ctx.train_path.exists()
    assert ctx.test_path.exists()
    assert ctx.sample_sub_path.exists()
    assert len(ctx.known_types) == 44
    assert ctx.known_types["proto"] == "category"
    assert ctx.known_types["dur"] == "numeric"


def test_load_regression_task_context():
    task_path = GIVEN_DATA_DIR / "Регрессия" / "task.md"
    assert task_path.exists(), f"Файл не найден: {task_path}"

    ctx = load_dataset_context(task_path)
    assert ctx.target_col == "site_eui"
    assert ctx.id_col == "id"
    assert ctx.task_type == "regression"
    assert ctx.metric == "rmse"
    assert ctx.train_path.exists()
    assert ctx.test_path.exists()
    assert ctx.sample_sub_path.exists()
    assert len(ctx.known_types) == 64
    assert ctx.known_types["floor_area"] == "numeric"
    assert ctx.known_types["building_class"] == "category"
