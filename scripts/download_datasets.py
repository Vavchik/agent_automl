"""
Скрипт загрузки эталонных датасетов для тестирования AutoML-пайплайна.

Сохраняет датасеты в папку `data/raw/`:
- titanic.csv (бинарная классификация, базовый smoke-тест)
- california_housing.csv (регрессия, базовый smoke-тест)
- telco_churn.csv (бинарная классификация: категории, пробелы в числах)
- dry_bean.csv (мультиклассовая классификация: 7 классов)
- ames_house_prices.csv (регрессия: 79 признаков, пропуски >90%)
- bike_sharing.csv (временной ряд: суточная сезонность, даты)
- diabetes_130_us.csv (групповой сплит: пациент patient_nbr)
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


CACHE_DIR = RAW_DATA_DIR.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_california_housing() -> Path:
    out_path = RAW_DATA_DIR / "california_housing.csv"
    if out_path.exists():
        print(f"✓ California Housing уже существует: {out_path}")
        return out_path
    print("⏳ Загрузка California Housing...")
    data = fetch_california_housing(data_home=str(CACHE_DIR), as_frame=True)
    df = data.frame
    df.to_csv(out_path, index=False)
    print(f"✓ Сохранен California Housing ({len(df)} строк) -> {out_path}")
    return out_path


def download_openml_dataset(name: str, data_id: int | str, filename: str, version: int | None = None) -> Path:
    out_path = RAW_DATA_DIR / filename
    if out_path.exists():
        print(f"✓ {name} уже существует: {out_path}")
        return out_path
    print(f"⏳ Загрузка {name} (OpenML id/name={data_id})...")
    try:
        if isinstance(data_id, int):
            data = fetch_openml(data_id=data_id, data_home=str(CACHE_DIR), as_frame=True, parser="auto")
        else:
            data = fetch_openml(data_id, version=version, data_home=str(CACHE_DIR), as_frame=True, parser="auto")
        df = data.frame
        df.to_csv(out_path, index=False)
        print(f"✓ Сохранен {name} ({len(df)} строк, {df.shape[1]} колонок) -> {out_path}")
        return out_path
    except Exception as e:
        print(f"❌ Ошибка загрузки {name}: {e}")
        return out_path


def main() -> None:
    print("=" * 60)
    print(f"📥 Загрузка эталонных датасетов в {RAW_DATA_DIR}")
    print("=" * 60)

    # 1. Smoke-тесты
    download_california_housing()
    download_openml_dataset("Titanic", "titanic", "titanic.csv", version=1)

    # 2. Бинарная с грязными типами
    download_openml_dataset("Telco Customer Churn", 42178, "telco_churn.csv")

    # 3. Мультикласс
    download_openml_dataset("Dry Bean", 42477, "dry_bean.csv")

    # 4. Сложная регрессия с пропусками
    download_openml_dataset("Ames House Prices", 42165, "ames_house_prices.csv")

    # 5. Временной сплит
    download_openml_dataset("Bike Sharing", 42712, "bike_sharing.csv")

    # 6. Групповой сплит (GroupKFold по пациентам)
    download_openml_dataset("Diabetes 130-US", 43896, "diabetes_130_us.csv")

    print("\n" + "=" * 60)
    print("🎉 Загрузка завершена! Список файлов в data/raw/:")
    for f in sorted(RAW_DATA_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name:25} ({size_kb:8.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
