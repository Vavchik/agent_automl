"""
Скрипт сквозного мета-анализа с использованием реальной локальной модели Qwen (Ollama).

Пайплайн:
1. Загрузка датасета (по умолчанию Titanic или указанный в аргументах).
2. Этап 1: Инференс типов и расчет математического профиля (ml_core).
3. Этап 2: Запрос к локальной LLM (Qwen 2.5 Coder 7B) через Ollama API.
4. Получение строгого JSON от Qwen и сохранение в data/artifacts/decision_<dataset>.json.
5. Вывод решения и обоснования модели.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from agent.llm import ask_llm, extract_json
from agent.meta_analyst import build_meta_analysis_prompt, get_meta_analysis_system_prompt
from ml_core.data_profiler import profile_data
from ml_core.schemas import ModelSelectionDecision
from ml_core.type_inference import infer_types

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_meta_analysis(dataset_path: Path, target: str, model_name: str = "qwen2.5-coder:7b") -> ModelSelectionDecision:
    dataset_name = dataset_path.stem
    print("=" * 65)
    print(f"🤖 Запуск Мета-Анализа с локальной моделью Qwen: {model_name}")
    print(f"📊 Датасет: {dataset_path.name} | Таргет: '{target}'")
    print("=" * 65)

    # 1. Загрузка
    raw_df = pd.read_csv(dataset_path)
    print(f"\n[Шаг 1] Исходные данные: {raw_df.shape[0]} строк, {raw_df.shape[1]} колонок.")

    # 2. Этап 1: Инференс типов
    print("[Шаг 2] Очистка и определение типов (type_inference)...")
    clean_df, column_types = infer_types(raw_df)

    # 3. Этап 1: Профилирование
    print("[Шаг 3] Сбор математического профиля (data_profiler)...")
    report = profile_data(clean_df, column_types, target=target)
    print(f"✓ Профиль собран: {len(report.column_types)} колонок, {len(report.target_correlation)} связей с таргетом.")

    # 4. Этап 2: Запрос к Qwen
    print(f"\n[Шаг 4] Отправка метаданных в локальный Qwen ({model_name})...")
    system_prompt = get_meta_analysis_system_prompt()
    user_prompt = build_meta_analysis_prompt(report)

    raw_response = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model_name,
        temperature=0.1,
    )
    print("✓ Ответ получен от Qwen!")

    # 5. Парсинг ответа
    parsed_json = extract_json(raw_response)
    decision = ModelSelectionDecision.from_dict(parsed_json)

    # 6. Сохранение артефакта
    output_json_path = ARTIFACTS_DIR / f"decision_{dataset_name}.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n[Шаг 5] Итоговый JSON сохранен -> {output_json_path}")
    print("-" * 65)
    print("📋 РЕШЕНИЕ QWEN:")
    print(f"  • Тип задачи:    {decision.task_type}")
    print(f"  • Черновик:      {decision.draft_model.algorithm} ({decision.draft_model.library_class})")
    print(f"  • Боевая модель: {decision.battle_model.algorithm} ({decision.battle_model.library_class})")
    print(f"  • Инструмент:    {decision.battle_model.tuning_tool}")
    print(f"  • Функция потерь:{decision.loss_function}")
    print(f"  • Метрика:       {decision.metric}")
    print(f"  • Обоснование Qwen:\n    {decision.reasoning}")
    print("=" * 65)

    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск мета-анализа с локальной LLM Qwen")
    parser.add_argument("--dataset", type=str, default="titanic.csv", help="Имя CSV-файла в data/raw/")
    parser.add_argument("--target", type=str, default="", help="Целевая колонка (если не указана, автодетект)")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Модель Ollama")

    args = parser.parse_args()
    csv_path = RAW_DIR / args.dataset
    if not csv_path.exists():
        print(f"❌ Файл {csv_path} не найден! Доступные датасеты в data/raw/:")
        for f in RAW_DIR.glob("*.csv"):
            print(f"  • {f.name}")
        sys.exit(1)

    target = args.target
    if not target:
        # Автодетект таргета под известные датасеты
        sample = pd.read_csv(csv_path, nrows=1)
        for col in ["survived", "Survived", "churn", "Churn", "MedHouseVal", "price", "count", "cnt", "target", "y"]:
            if col in sample.columns:
                target = col
                break
        if not target:
            target = sample.columns[-1]

    run_meta_analysis(csv_path, target=target, model_name=args.model)


if __name__ == "__main__":
    main()
