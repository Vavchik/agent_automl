"""
Единая точка входа для платформы соревнования (Alfa Hack AutoML).

Интерфейс запуска, зафиксированный платформой:
    python solution.py task.md
    python solution.py /path/to/task.md

Архитектура сквозного решения (Dual-Brain):
1. Мета-загрузка: парсинг task.md, data_dictionary.csv, sample_submission.csv.
2. Профилирование и Мета-анализ: определение типов, выбросов, выбор боевой модели (LLM Qwen).
3. Failsafe Fast Baseline (Мин. 0 – 5): 5-Fold CV + калибровка + мгновенная запись submission.csv.
4. AutoResearch Цикл признаков (Мин. 5 – 45): генерация гипотез через LLM, валидация в песочнице,
   принятие/откат и рефлексия.
5. Тяжелая артиллерия (Tier 2 Upgrade): мультимодельный стэкинг AutoGluon TabularPredictor на 8 CPU.
6. Submission Guardian: гарантия строгой валидности файла предсказаний перед завершением.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

# Добавляем корень проекта в PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.autoresearch import AutoResearchLoop
from agent.orchestrator import AutoMLOrchestrator
from ml_core.automl_engine import AutoMLEnsembleEngine
from ml_core.dataset_loader import DatasetContext, load_dataset_context
from ml_core.evaluator import is_metric_better
from ml_core.logger import setup_logger
from ml_core.postprocessor import apply_calibrated_predictions, optimize_class_weights, postprocess_regression
from ml_core.submission_guard import save_submission_safely
from ml_core.trainer import CrossValidatedTrainer


def run_solution(task_md_path: Path | str, time_limit_sec: float = 9000.0) -> Path:
    """
    Запускает полный автономный пайплайн решения задачи соревнования.
    """
    start_time = time.time()
    task_path = Path(task_md_path).resolve()

    # ----------------------------------------------------------------------- #
    # Шаг 1. Мета-загрузка контекста задачи
    # ----------------------------------------------------------------------- #
    ctx = load_dataset_context(task_path)
    logger = setup_logger(task_dir=ctx.task_dir)

    logger.info("=" * 70)
    logger.info(f"🚀 Alfa Hack AutoML Agent: Старт решения задачи '{task_path.name}'")
    logger.info("=" * 70)
    logger.info(f"📁 Каталог задачи:      {ctx.task_dir}")
    print(f"🎯 Целевая колонка:    '{ctx.target_col}' | ID: '{ctx.id_col}'")
    print(f"📊 Тип задачи:         {ctx.task_type} | Метрика: {ctx.metric}")
    print(f"📑 Обучающая выборка:  {ctx.train_path.name} | Тестовая: {ctx.test_path.name}")
    print(f"📖 Словарь признаков:  {ctx.dict_path.name if ctx.dict_path else 'не найден'} ({len(ctx.known_types)} типов)")

    # ----------------------------------------------------------------------- #
    # Шаг 2. Загрузка данных и очистка
    # ----------------------------------------------------------------------- #
    try:
        import pandas as pd
        train_df = pd.read_csv(ctx.train_path)
        test_df = pd.read_csv(ctx.test_path)
        print(f"\n✓ Данные загружены: train={train_df.shape}, test={test_df.shape}")
    except Exception as e:
        print(f"❌ Критическая ошибка при загрузке данных через pandas: {e}")
        raise

    # ----------------------------------------------------------------------- #
    # Шаг 3. Профилирование и Мета-анализ задачи (LLM Qwen)
    # ----------------------------------------------------------------------- #
    orchestrator = AutoMLOrchestrator(
        target=ctx.target_col,
        id_col=ctx.id_col,
        known_types=ctx.known_types,
        task_description=ctx.task_text,
        metric_hint=ctx.metric,
    )

    print("\n[Стадия 1/5] Инференс типов и профилирование данных...")
    report = orchestrator.run_stage_profile(train_df)
    clean_train_df = orchestrator.clean_df if orchestrator.clean_df is not None else train_df

    from ml_core.type_inference import infer_types
    clean_test_df, _ = infer_types(test_df, known_types=ctx.known_types)

    print("\n[Стадия 2/5] Мета-анализ задачи и выбор архитектуры (LLM Qwen)...")
    decision = orchestrator.run_stage_meta_analysis()
    chosen_algo = decision.battle_model.algorithm if decision and decision.battle_model else "LightGBM"
    print(f"   • Выбранный алгоритм:   {chosen_algo}")
    print(f"   • Целевая функция потерь: {decision.loss_function}")
    print(f"   • Обоснование агента:     {decision.reasoning[:120]}...")

    # ----------------------------------------------------------------------- #
    # Шаг 4. Failsafe Fast Baseline (Мин. 0 – 5): 5-Fold CV + submission.csv v0
    # ----------------------------------------------------------------------- #
    print("\n[Стадия 3/5] Обучение Failsafe Baseline (5-Fold CV) и фиксация submission.csv v0...")
    try:
        import lightgbm
        import catboost
        effective_algo = "Blend"
    except ImportError:
        effective_algo = chosen_algo

    print(f"   • Выбранный движок обучения: {effective_algo} (LLM выбор: {chosen_algo})")

    trainer = CrossValidatedTrainer(
        task_type=ctx.task_type,
        metric=ctx.metric,
        algorithm=effective_algo,
        random_state=42,
    )

    loop = AutoResearchLoop(
        ctx=ctx,
        trainer=trainer,
        max_iterations=4,
        time_budget_sec=min(1800.0, time_limit_sec * 0.4),
        min_improvement_delta=1e-4,
    )

    base_score, sub_path = loop.run_baseline(clean_train_df, clean_test_df)
    print(f"   ⭐ Failsafe Baseline CV Score: {base_score:.5f}")
    print(f"   💾 submission.csv сохранен:    {sub_path} ({sub_path.stat().st_size} байт)")

    # ----------------------------------------------------------------------- #
    # Шаг 5. AutoResearch Цикл признаков (Feature Engineering + Reflection)
    # ----------------------------------------------------------------------- #
    print("\n[Стадия 4/5] Запуск автономного AutoResearch цикла генерации признаков...")
    research_res = loop.run_research(clean_train_df, clean_test_df)

    print(f"\n   • Базовый CV скор:  {research_res.baseline_score:.5f}")
    print(f"   • Итоговый CV скор: {research_res.best_score:.5f}")
    print(f"   • Принято гипотез:  {len(research_res.accepted_records)}")
    for acc in research_res.accepted_records:
        print(f"     ✓ #{acc.iteration}: {acc.description} (прирост: {acc.delta:+.5f})")

    best_train_features = research_res.best_train_df
    best_test_features = research_res.best_test_df
    current_best_score = research_res.best_score

    # ----------------------------------------------------------------------- #
    # Шаг 6. Тяжелая артиллерия (Tier 2 Upgrade: AutoGluon Multi-Model Ensemble)
    # ----------------------------------------------------------------------- #
    elapsed_so_far = time.time() - start_time
    remaining_time = time_limit_sec - elapsed_so_far

    print("\n[Стадия 5/5] Проверка возможности запуска глубокого ансамбля (Tier 2: AutoGluon)...")
    ensemble_engine = AutoMLEnsembleEngine(
        task_type=ctx.task_type,
        metric=ctx.metric,
        time_limit_sec=int(min(1800, remaining_time - 300)),
        num_cpus=8,
    )

    if ensemble_engine.is_available() and remaining_time > 600:
        print(f"   • AutoGluon доступен! Запуск обучения на {ensemble_engine.num_cpus} CPU...")
        ag_score, ag_preds = ensemble_engine.fit_and_predict(
            train_df=best_train_features,
            test_df=best_test_features,
            target_col=ctx.target_col,
            id_col=ctx.id_col,
        )

        if ag_score is not None and ag_preds is not None:
            # Проверяем, превзошел ли ансамбль наш лучший CatBoost/LightGBM
            if is_metric_better(ag_score, current_best_score, metric=ctx.metric, min_delta=1e-4):
                print(f"   🎉 AutoGluon превзошел CV модели ({ag_score:.5f} vs {current_best_score:.5f})!")
                # Пост-процессинг предсказаний
                is_numeric = "class" not in ctx.task_type.lower()
                if is_numeric:
                    final_preds = postprocess_regression(ag_preds, lower_bound=0.0)
                else:
                    final_preds = ag_preds

                # Извлекаем тестовые ID
                test_ids = list(best_test_features[ctx.id_col]) if ctx.id_col in best_test_features.columns else list(range(len(best_test_features)))
                save_submission_safely(
                    ids=test_ids,
                    preds=final_preds,
                    sample_sub_path=ctx.sample_sub_path,
                    output_path=sub_path,
                    is_numeric_target=is_numeric,
                )
                print(f"   💾 submission.csv обновлен предсказаниями AutoGluon ансамбля!")
                current_best_score = ag_score
            else:
                print(f"   • AutoGluon скор ({ag_score:.5f}) не превысил наш лучший результат ({current_best_score:.5f}). Сохраняем решение AutoResearch.")
    else:
        print("   • Пропуск AutoGluon (библиотека не установлена или лимит времени исчерпан).")

    # ----------------------------------------------------------------------- #
    # Финал: Итоговый отчет
    # ----------------------------------------------------------------------- #
    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print("🏁 Решение успешно завершено!")
    print(f"📁 Итоговый файл:     {sub_path} ({sub_path.stat().st_size} байт)")
    print(f"🏆 Лучший CV Score:    {current_best_score:.5f} ({ctx.metric})")
    print(f"⏱ Общее время работы: {total_time} секунд")
    print("=" * 70)
    return sub_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Alfa Hack AutoML Agent Solution")
    parser.add_argument(
        "task_path",
        nargs="?",
        default="task.md",
        help="Путь к файлу task.md (по умолчанию task.md в текущей папке)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=9000.0,
        help="Максимальный бюджет времени в секундах (по умолчанию 9000 с / 2.5 часа)",
    )
    args = parser.parse_args()
    run_solution(args.task_path, time_limit_sec=args.time_limit)


if __name__ == "__main__":
    main()
