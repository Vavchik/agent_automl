"""
Модуль интеллектуальной загрузки и разбора материалов задачи:
- Поиск train.csv, test.csv, data_dictionary.csv, sample_submission.csv в каталоге задачи.
- Извлечение ID и таргета из data_dictionary.csv и sample_submission.csv.
- Парсинг описания задачи и целевой метрики из task.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Optional


@dataclass
class DatasetContext:
    task_dir: Path
    task_md_path: Path
    task_text: str
    train_path: Path
    test_path: Path
    dict_path: Optional[Path]
    sample_sub_path: Path
    target_col: str
    id_col: str
    task_type: str                  # "regression" | "multiclass_classification" | "binary_classification"
    metric: str                     # "rmse" | "macro_f1" | "roc_auc"
    known_types: dict[str, str] = field(default_factory=dict)
    feature_descriptions: dict[str, str] = field(default_factory=dict)


def _find_file(base_dir: Path, filename: str) -> Optional[Path]:
    """Ищет файл в каталоге или вложенных папках (например, participants/)."""
    direct = base_dir / filename
    if direct.exists():
        return direct
    matches = list(base_dir.rglob(filename))
    return matches[0] if matches else None


def load_dataset_context(task_md_path: Path | str) -> DatasetContext:
    """
    Анализирует каталог задачи и возвращает полный контекст датасета.
    """
    task_path = Path(task_md_path).resolve()
    if not task_path.exists():
        raise FileNotFoundError(f"Файл задачи не найден: {task_path}")

    task_dir = task_path.parent
    task_text = task_path.read_text(encoding="utf-8")

    # 1. Поиск обязательных файлов
    train_path = _find_file(task_dir, "train.csv")
    test_path = _find_file(task_dir, "test.csv")
    sample_sub_path = _find_file(task_dir, "sample_submission.csv")
    dict_path = _find_file(task_dir, "data_dictionary.csv")

    if not train_path or not test_path or not sample_sub_path:
        raise FileNotFoundError(
            f"Не найдены обязательные файлы в каталоге {task_dir}: "
            f"train={bool(train_path)}, test={bool(test_path)}, sample_sub={bool(sample_sub_path)}"
        )

    # 2. Анализ sample_submission.csv
    with open(sample_sub_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_headers = next(reader)
    if len(sub_headers) < 2:
        raise ValueError(f"sample_submission.csv должен содержать минимум 2 колонки: {sub_headers}")

    id_col = sub_headers[0]
    target_col = sub_headers[1]

    # 3. Анализ data_dictionary.csv (если есть)
    known_types: dict[str, str] = {}
    feature_descriptions: dict[str, str] = {}

    if dict_path and dict_path.exists():
        with open(dict_path, mode="r", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                feat = row.get("feature") or row.get("column") or row.get("name")
                if not feat:
                    continue
                role = str(row.get("role", "")).lower().strip()
                dtype = str(row.get("dtype", "")).lower().strip()
                desc = str(row.get("description_ru") or row.get("description", "")).strip()

                if role == "target":
                    target_col = feat
                elif role in ("id", "identifier"):
                    id_col = feat

                if dtype:
                    known_types[feat] = dtype
                if desc:
                    feature_descriptions[feat] = desc

    # 4. Определение метрики и типа задачи по тексту task.md
    task_text_lower = task_text.lower()

    # Метрика
    if "macro f1" in task_text_lower or "f1_macro" in task_text_lower or "macro-f1" in task_text_lower:
        metric = "macro_f1"
    elif "rmse" in task_text_lower or "root mean squared" in task_text_lower:
        metric = "rmse"
    elif "roc_auc" in task_text_lower or "roc-auc" in task_text_lower or "auc" in task_text_lower:
        metric = "roc_auc"
    elif "accuracy" in task_text_lower:
        metric = "accuracy"
    else:
        metric = "rmse" if "регресс" in task_text_lower else "macro_f1"

    # Тип задачи
    if "многоклассов" in task_text_lower or "multiclass" in task_text_lower or metric == "macro_f1":
        task_type = "multiclass_classification"
    elif "регресс" in task_text_lower or "regression" in task_text_lower or metric == "rmse":
        task_type = "regression"
    elif "бинарн" in task_text_lower or "классификац" in task_text_lower:
        task_type = "binary_classification"
    else:
        task_type = "regression" if metric in ("rmse", "mae") else "multiclass_classification"

    return DatasetContext(
        task_dir=task_dir,
        task_md_path=task_path,
        task_text=task_text,
        train_path=train_path,
        test_path=test_path,
        dict_path=dict_path,
        sample_sub_path=sample_sub_path,
        target_col=target_col,
        id_col=id_col,
        task_type=task_type,
        metric=metric,
        known_types=known_types,
        feature_descriptions=feature_descriptions,
    )
