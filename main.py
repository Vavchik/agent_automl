"""
Точка входа в приложение AutoML Agent.

Позволяет запустить сквозной пайплайн на любом датасете:
1. Загрузка данных (из data/raw/ или по указанному пути к CSV).
2. Очистка типичного мусора и вывод типов (ml_core.type_inference).
3. Сбор математического профиля и корреляций с таргетом (ml_core.data_profiler).
4. Мета-анализ: выбор моделей и метрик локальной LLM Qwen (agent.meta_analyst).
5. Сохранение результирующего JSON-файла в data/artifacts/decision_<dataset>.json.
6. Разбиение данных на train/val/test выборки (ml_core.splitter).

Примеры использования:
    python main.py
    python main.py --dataset titanic.csv
    python main.py --dataset california_housing.csv --target MedHouseVal
    python main.py --dataset telco_churn.csv --target Churn
    python main.py --dataset data/raw/bike_sharing.csv --time-col dteday
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from agent import AutoMLOrchestrator

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"


# Частые имена целевых колонок для автоматического определения
COMMON_TARGET_NAMES = [
    "survived", "Survived",
    "churn", "Churn",
    "MedHouseVal", "medhouseval",
    "SalePrice", "price", "Price",
    "charges", "Charges",
    "target", "Target", "class", "Class", "y", "label",
    "count", "cnt",
    "readmitted",
]


def list_available_datasets() -> list[Path]:
    """Возвращает список доступных CSV файлов в папке data/raw/."""
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.csv"))


def autodetect_target(df: pd.DataFrame) -> str:
    """Пытается угадать целевую колонку по частым именам или берет последнюю."""
    for candidate in COMMON_TARGET_NAMES:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Находит файл датасета по прямому пути или внутри data/raw/."""
    direct_path = Path(dataset_arg)
    if direct_path.exists():
        return direct_path

    in_raw = RAW_DIR / dataset_arg
    if in_raw.exists():
        return in_raw

    # Пробуем добавить расширение .csv
    in_raw_csv = RAW_DIR / f"{dataset_arg}.csv"
    if in_raw_csv.exists():
        return in_raw_csv

    available = list_available_datasets()
    avail_str = "\n".join([f"  • {f.name}" for f in available]) if available else "  (папка пуста)"
    raise FileNotFoundError(
        f"Датасет '{dataset_arg}' не найден!\n"
        f"Доступные датасеты в data/raw/:\n{avail_str}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoML Agent: автономный анализ датасета и выбор ML-архитектуры"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="titanic.csv",
        help="Имя файла в data/raw/ (например, titanic.csv, california_housing.csv) или прямой путь к CSV",
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        default="",
        help="Имя целевой переменной (таргета). Если не указано — определяется автоматически",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Имя колонки времени (если есть временная ось)",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Имя колонки групп (например user_id, patient_nbr)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки (по умолчанию 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Доля валидационной выборки (по умолчанию 0.1)",
    )
    parser.add_argument(
        "--target-metric-threshold",
        type=float,
        default=0.0,
        help="Минимально допустимый порог метрики (если ниже — эскалация в Tier 2/3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Создаём папку артефактов лениво (только при запуске, не при импорте)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🤖 Запуск AutoML Agent Pipeline")
    print("=" * 65)

    # 1. Поиск и загрузка датасета
    csv_path = resolve_dataset_path(args.dataset)
    dataset_name = csv_path.stem
    raw_df = pd.read_csv(csv_path)

    # 2. Определение таргета
    target = args.target if args.target else autodetect_target(raw_df)
    if target not in raw_df.columns:
        raise KeyError(
            f"Таргет '{target}' не найден в датасете. Доступные колонки: {list(raw_df.columns)}"
        )

    print(f"\n📁 1. Загружен датасет: {csv_path.name}")
    print(f"   • Путь: {csv_path}")
    print(f"   • Размерность: {raw_df.shape[0]} строк, {raw_df.shape[1]} колонок")
    print(f"   • Целевая переменная: '{target}' {'(автодетект)' if not args.target else ''}")
    if args.time_col:
        print(f"   • Колонка времени: '{args.time_col}'")
    if args.group_col:
        print(f"   • Колонка групп:   '{args.group_col}'")

    # 3. Инициализация оркестратора
    orchestrator = AutoMLOrchestrator(
        target=target,
        time_col=args.time_col,
        group_col=args.group_col,
    )

    # 4. Стадия 1: Инференс типов и профилирование
    print("\n🧹 2. Стадия 1: Очистка мусора, инференс типов и профилирование...")
    report = orchestrator.run_stage_profile(raw_df)
    print("   Типы колонок:")
    for col, ctype in report.column_types.items():
        mark = " [TARGET]" if col == target else ""
        print(f"     - {col:22} -> {ctype}{mark}")

    # 5. Стадия 2: Мета-анализ с локальной моделью (Qwen / Fallback)
    print("\n🧠 3. Стадия 2: Мета-анализ с моделью Qwen (выбор архитектур, лосса и метрики)...")
    decision = orchestrator.run_stage_meta_analysis()

    # 6. Сохранение результирующего JSON-файла
    output_json_path = ARTIFACTS_DIR / f"decision_{dataset_name}.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n💾 Результирующий JSON сохранен:")
    print(f"   👉 {output_json_path}")
    print("-" * 65)
    print("📋 РЕШЕНИЕ АГЕНТА:")
    print(f"   • Тип задачи:     {decision.task_type}")
    print(f"   • Черновик:       {decision.draft_model.algorithm} ({decision.draft_model.library_class})")
    print(f"   • Боевая модель:  {decision.battle_model.algorithm} ({decision.battle_model.library_class})")
    print(f"   • Тюнинг:         {decision.battle_model.tuning_tool}")
    print(f"   • Функция потерь: {decision.loss_function}")
    print(f"   • Метрика:        {decision.metric}")
    if decision.reasoning:
        print(f"   • Обоснование:\n     {decision.reasoning}")

    # 7. Стадия 3: Разбиение выборки
    print("\n✂️ 4. Стадия 3: Разбиение данных на train / val / test...")
    splits = orchestrator.run_stage_split(test_size=args.test_size, val_size=args.val_size)
    summary = splits.summary()
    print(f"   • Стратегия сплита: {summary['strategy']}")
    print(f"   • Train (обучение): {summary['train_shape']}")
    print(f"   • Val (валидация):  {summary['val_shape']}")
    print(f"   • Test (тест):      {summary['test_shape']}")

    print("\n" + "=" * 65)
    print(f"✅ Готово! Проверьте файл решения: data/artifacts/decision_{dataset_name}.json")
    print("=" * 65)


if __name__ == "__main__":
    main()