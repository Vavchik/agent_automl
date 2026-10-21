"""
Модуль валидации и атомарного сохранения сабмита (Submission Guardian).

Гарантирует:
1. Соответствие колонок и их порядка файлу sample_submission.csv.
2. Совпадение количества строк и построчного порядка ID.
3. Отсутствие NaN, null, бесконечностей (Inf) и пустых строк.
4. Атомарную запись через временный файл, исключающую повреждение файла при аварийной остановке контейнера.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None


class SubmissionValidationError(ValueError):
    """Исключение при нарушении формата файла сабмита."""
    pass


def _normalize_id(val: Any) -> str:
    s = str(val).strip()
    if s.endswith(".0"):
        try:
            int_val = int(float(s))
            if str(float(int_val)) == s:
                return str(int_val)
        except (ValueError, OverflowError):
            pass
    return s


def validate_submission_data(
    sub_ids: Sequence[Any],
    sub_preds: Sequence[Any],
    sample_ids: Sequence[Any],
    target_name: str = "target",
    is_numeric_target: bool = False,
) -> Tuple[bool, str]:
    """
    Проверяет корректность предсказаний относительно sample_submission.
    Работает со списками, numpy arrays или pandas Series.
    """
    n_sub = len(sub_ids)
    n_sample = len(sample_ids)

    if n_sub != n_sample:
        return False, f"Несовпадение числа строк: в сабмите {n_sub}, ожидалось {n_sample}"

    if len(sub_preds) != n_sub:
        return False, f"Число предсказаний ({len(sub_preds)}) не равно числу ID ({n_sub})"

    # 1. Проверка построчного соответствия ID с нормализацией float-индексов
    for idx, (s_id, exp_id) in enumerate(zip(sub_ids, sample_ids)):
        if _normalize_id(s_id) != _normalize_id(exp_id):
            return False, f"Несовпадение ID в строке {idx}: '{s_id}' != '{exp_id}'"

    # 2. Проверка валидности предсказаний
    for idx, val in enumerate(sub_preds):
        if val is None:
            return False, f"Обнаружен None в предсказании строки {idx}"
        if isinstance(val, str) and val.strip() == "":
            return False, f"Обнаружена пустая строка в предсказании строки {idx}"

        if is_numeric_target or isinstance(val, (int, float)):
            try:
                num_val = float(val)
                if math.isnan(num_val):
                    return False, f"Обнаружен NaN в строке {idx}"
                if math.isinf(num_val):
                    return False, f"Обнаружен Infinity в строке {idx}"
            except (ValueError, TypeError):
                return False, f"Нечисловое значение '{val}' для числового таргета в строке {idx}"

    return True, "OK"


def save_submission_safely(
    ids: Sequence[Any],
    preds: Sequence[Any],
    sample_sub_path: Union[Path, str],
    output_path: Union[Path, str],
    is_numeric_target: bool = False,
) -> Path:
    """
    Валидирует и атомарно записывает CSV файл submission.csv.
    """
    sample_path = Path(sample_sub_path).resolve()
    out_path = Path(output_path).resolve()

    # Считываем sample_submission для валидации
    import csv
    with open(sample_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_rows = list(reader)

    if len(header) < 2:
        raise SubmissionValidationError(f"sample_submission содержит меньше 2 колонок: {header}")

    id_col = header[0]
    target_col = header[1]
    sample_ids = [row[0] for row in sample_rows]

    # Валидация
    is_valid, msg = validate_submission_data(
        sub_ids=ids,
        sub_preds=preds,
        sample_ids=sample_ids,
        target_name=target_col,
        is_numeric_target=is_numeric_target,
    )
    if not is_valid:
        raise SubmissionValidationError(f"Ошибка валидации сабмита: {msg}")

    # Атомарная запись: пишем эталонные sample_ids и предсказания во временный файл
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([id_col, target_col])
        for s_id, pred in zip(sample_ids, preds):
            writer.writerow([s_id, pred])

    # Атомарный rename
    tmp_path.replace(out_path)
    return out_path
