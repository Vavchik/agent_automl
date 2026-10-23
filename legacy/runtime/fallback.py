"""Р1. Аварийный сабмит: срабатывает, если пайплайн упал до записи нормального ответа.

Не зависит от остального кода: только pandas и файлы задачи.
Регрессия -> медиана таргета, классификация -> самый частый класс,
вероятности -> доля положительного класса. Если не вышло и это — копия sample_submission.
"""
from pathlib import Path

import pandas as pd


def _find(task_dir: Path, name: str) -> Path | None:
    direct = task_dir / name
    if direct.exists():
        return direct
    found = sorted(task_dir.rglob(name))
    return found[0] if found else None


def write_fallback(task_md: Path) -> Path:
    task_dir = Path(task_md).resolve().parent
    out = task_dir / "submission.csv"
    sample_path = _find(task_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    try:
        test = pd.read_csv(_find(task_dir, "test.csv"))
        train = pd.read_csv(_find(task_dir, "train.csv"))
        id_col = sample.columns[0]
        sub = pd.DataFrame({id_col: test[id_col]})
        for col in sample.columns[1:]:
            if col in train.columns:
                target = train[col].dropna()
                is_float_sample = pd.api.types.is_float_dtype(sample[col])
                if pd.api.types.is_numeric_dtype(target) and (target.nunique() > 20 or is_float_sample):
                    sub[col] = float(target.median())
                else:
                    sub[col] = target.mode().iloc[0]
            else:
                sub[col] = sample[col].iloc[0]
        sub.to_csv(out, index=False)
    except Exception:
        sample.to_csv(out, index=False)
    return out
