"""
Модуль автономного исследовательского цикла признаков (AutoResearchLoop).

Реализует:
1. Обучение быстрого бейзлайна (Baseline) с немедленным сохранением submission.csv.
2. Итеративный цикл генерации гипотез признаков (LLM Qwen или Kaggle-эвристики).
3. Экспериментальную валидацию в песочнице на 5-Fold кросс-валидации.
4. Принятие (Accept) или откат (Rollback) гипотез с динамическим обновлением submission.csv.
5. Рефлексию (Reflection) на ошибках и падениях метрики через ExperimentMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Optional, Sequence
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from agent.feature_engineer import execute_feature_transform, generate_feature_code
from agent.memory import ExperimentMemory, ExperimentRecord
from ml_core.dataset_loader import DatasetContext
from ml_core.evaluator import evaluate_oof, is_metric_better
from ml_core.postprocessor import apply_calibrated_predictions, optimize_class_weights, postprocess_regression
from ml_core.submission_guard import save_submission_safely
from ml_core.trainer import CrossValidatedTrainer, TrainingResult

logger = logging.getLogger("agent.autoresearch")


@dataclass
class AutoResearchResult:
    """Итоговый результат исследовательского цикла."""
    baseline_score: float
    best_score: float
    best_train_df: Any
    best_test_df: Any
    accepted_records: list[ExperimentRecord] = field(default_factory=list)
    memory: Optional[ExperimentMemory] = None
    sub_path: Optional[Path] = None


class AutoResearchLoop:
    """
    Автономный агент-исследователь признаков для табличных данных.
    """

    def __init__(
        self,
        ctx: DatasetContext,
        memory: Optional[ExperimentMemory] = None,
        trainer: Optional[CrossValidatedTrainer] = None,
        max_iterations: int = 3,
        time_budget_sec: float = 1200.0,
        min_improvement_delta: float = 1e-4,
    ):
        self.ctx = ctx
        self.is_classification = "class" in ctx.task_type.lower()
        self.metric = ctx.metric
        self.higher_is_better = not (self.metric.lower() in ("rmse", "root_mean_squared_error", "mae", "logloss"))
        self.memory = memory or ExperimentMemory(metric=self.metric, higher_is_better=self.higher_is_better)
        self.trainer = trainer or CrossValidatedTrainer(
            task_type=ctx.task_type,
            metric=self.metric,
            algorithm="LightGBM",
        )
        self.max_iterations = max_iterations
        self.time_budget_sec = time_budget_sec
        self.min_improvement_delta = min_improvement_delta

        self.best_score: Optional[float] = None
        self.baseline_score: Optional[float] = None
        self.best_train: Any = None
        self.best_test: Any = None
        self.latest_top_features: list[tuple[str, float]] = []

    def _extract_xy(
        self,
        train_data: Any,
        test_data: Optional[Any] = None,
    ) -> tuple[Any, Any, Optional[Any], list[Any]]:
        """Извлекает матрицы признаков X, таргета y и тестовые ID."""
        target_col = self.ctx.target_col
        id_col = self.ctx.id_col

        if pd is not None and isinstance(train_data, pd.DataFrame):
            drop_tr = [col for col in [target_col, id_col] if col in train_data.columns]
            X_train = train_data.drop(columns=drop_tr)
            y_train = train_data[target_col].to_numpy()

            if test_data is not None:
                drop_te = [col for col in [target_col, id_col] if col in test_data.columns]
                X_test = test_data.drop(columns=drop_te)
                test_ids = list(test_data[id_col]) if id_col in test_data.columns else list(range(len(test_data)))
            else:
                X_test = None
                test_ids = []
            return X_train, y_train, X_test, test_ids

        # Pure arrays fallback
        y_train = np.asarray(train_data[target_col])
        X_train = np.asarray(train_data["features"])
        X_test = np.asarray(test_data["features"]) if test_data is not None else None
        test_ids = list(test_data["ids"]) if test_data is not None else []
        return X_train, y_train, X_test, test_ids

    def _postprocess_and_evaluate(
        self,
        train_res: TrainingResult,
        y_train: np.ndarray,
    ) -> tuple[float, list[Any], list[Any]]:
        """
        Применяет калибровку под метрику и возвращает (score, postprocessed_oof, postprocessed_test).
        """
        if self.is_classification and train_res.oof_probs is not None:
            # Оптимизируем веса классов для максимизации Macro F1
            weights = optimize_class_weights(
                oof_probs=train_res.oof_probs,
                y_true=y_train,
                classes=train_res.classes,
                metric=self.metric,
            )
            calibrated_oof = apply_calibrated_predictions(
                probs=train_res.oof_probs,
                classes=train_res.classes,
                weights=weights,
            )
            score = evaluate_oof(
                y_true=y_train,
                y_oof_pred=calibrated_oof,
                metric=self.metric,
            )
            calibrated_test = (
                apply_calibrated_predictions(
                    probs=train_res.test_probs,
                    classes=train_res.classes,
                    weights=weights,
                )
                if train_res.test_probs is not None
                else []
            )
            return score, calibrated_oof, calibrated_test

        elif not self.is_classification:
            # Регрессия: клиппинг неотрицательности для энергопотребления
            post_oof = postprocess_regression(train_res.oof_preds, lower_bound=0.0)
            score = evaluate_oof(
                y_true=y_train,
                y_oof_pred=post_oof,
                metric=self.metric,
            )
            post_test = (
                postprocess_regression(train_res.test_preds, lower_bound=0.0)
                if len(train_res.test_preds) > 0
                else []
            )
            return score, post_oof, post_test

        return train_res.oof_score, list(train_res.oof_preds), list(train_res.test_preds)

    def run_baseline(
        self,
        train_df: Any,
        test_df: Any,
    ) -> tuple[float, Path]:
        """
        Этап 3: Быстро обучает бейзлайн (5-Fold CV) и немедленно сохраняет submission.csv v0.
        """
        logger.info("Запуск вычисления базовой модели (Baseline 5-Fold CV)...")
        X_train, y_train, X_test, test_ids = self._extract_xy(train_df, test_df)

        train_res = self.trainer.train_cv(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        score, _, test_preds = self._postprocess_and_evaluate(train_res, y_train)

        # Сохраняем submission.csv немедленно через SubmissionGuard
        sub_path = self.ctx.task_dir / "submission.csv"
        is_numeric = not self.is_classification
        save_submission_safely(
            ids=test_ids,
            preds=test_preds,
            sample_sub_path=self.ctx.sample_sub_path,
            output_path=sub_path,
            is_numeric_target=is_numeric,
        )
        logger.info(f"✓ Failsafe Baseline сохранен в {sub_path} со скором {score:.5f}")

        self.baseline_score = score
        self.best_score = score
        self.best_train = train_df
        self.best_test = test_df

        # Сохраняем топ-10 важных признаков
        self.latest_top_features = list(train_res.feature_importances.items())[:10]

        # Регистрируем бейзлайн в памяти
        self.memory.log_experiment(
            iteration=0,
            name="baseline_model",
            description="Базовая 5-Fold модель со стандартным предобработчиком",
            code="# baseline",
            score=score,
            baseline_score=score,
            status="accepted",
        )
        return score, sub_path

    def run_research(
        self,
        train_df: Any,
        test_df: Any,
    ) -> AutoResearchResult:
        """
        Этап 4: Автономный цикл генерации и отбора признаков.
        """
        start_time = time.time()
        sub_path = self.ctx.task_dir / "submission.csv"

        # 1. Стартуем с бейзлайна, если еще не был запущен
        if self.baseline_score is None:
            _, sub_path = self.run_baseline(train_df, test_df)

        current_train = self.best_train
        current_test = self.best_test
        current_score = self.best_score

        # Шаг 2a. Проверка Grandmaster GroupBy когортных признаков (быстрый буст)
        try:
            from ml_core.feature_recipes import apply_cohort_aggregations
            cohort_train, cohort_test, created = apply_cohort_aggregations(
                current_train,
                current_test,
                target_col=self.ctx.target_col,
                id_col=self.ctx.id_col,
                max_new_features=8,
            )
            if created and len(created) > 0:
                logger.info(f"Тестирование соревновательного рецепта когортных признаков ({len(created)} фичей)...")
                X_c_tr, y_c_tr, X_c_te, test_ids = self._extract_xy(cohort_train, cohort_test)
                c_train_res = self.trainer.train_cv(
                    X_train=X_c_tr,
                    y_train=y_c_tr,
                    X_test=X_c_te,
                )
                c_score, _, c_test_preds = self._postprocess_and_evaluate(c_train_res, y_c_tr)

                c_better = is_metric_better(
                    new_score=c_score,
                    base_score=current_score,
                    metric=self.metric,
                    min_delta=self.min_improvement_delta,
                )
                if c_better:
                    logger.info(f"🎉 ПРИНЯТО! Рецепт когортных агрегаций улучшил скор: {current_score:.5f} -> {c_score:.5f}")
                    current_score = c_score
                    current_train = cohort_train
                    current_test = cohort_test
                    self.best_score = c_score
                    self.best_train = cohort_train
                    self.best_test = cohort_test
                    self.latest_top_features = list(c_train_res.feature_importances.items())[:10]

                    save_submission_safely(
                        ids=test_ids,
                        preds=c_test_preds,
                        sample_sub_path=self.ctx.sample_sub_path,
                        output_path=sub_path,
                        is_numeric_target=not self.is_classification,
                    )
                    self.memory.log_experiment(
                        iteration=1,
                        name="cohort_groupby_recipe",
                        description=f"Grandmaster GroupBy когортные агрегации ({', '.join(created[:3])})",
                        code="# apply_cohort_aggregations",
                        score=c_score,
                        baseline_score=self.baseline_score,
                        status="accepted",
                    )
                else:
                    logger.info(f"Когортные признаки отклонены ({c_score:.5f} не превысил {current_score:.5f})")
        except Exception as e:
            logger.warning(f"Ошибка проверки когортных признаков: {e}")

        # 2. Исследовательский цикл на базе LLM
        for iteration in range(1, self.max_iterations + 1):
            elapsed = time.time() - start_time
            time_margin = min(180.0, self.time_budget_sec * 0.1)
            if (self.time_budget_sec - elapsed) < time_margin:
                logger.warning("Остаток времени исчерпан. Завершаем AutoResearch цикл.")
                break

            iter_start = time.time()
            logger.info(f"\n--- [AutoResearch] Итерация #{iteration}/{self.max_iterations} ---")

            # Генерация гипотезы и кода
            reflection_text = self.memory.format_history_for_llm(max_records=4)
            code, desc = generate_feature_code(
                task_description=self.ctx.task_text,
                feature_dictionary=self.ctx.feature_descriptions,
                metric_name=self.metric,
                current_best_score=current_score,
                top_features=self.latest_top_features,
                reflection_history=reflection_text,
                iteration=iteration,
            )

            # Исполнение кода трансформации с изоляцией таргета
            cand_train, cand_test, err = execute_feature_transform(
                code, current_train, current_test, target_col=self.ctx.target_col
            )
            if err:
                logger.warning(f"Итерация #{iteration} завершилась ошибкой кода: {err}")
                self.memory.log_experiment(
                    iteration=iteration,
                    name=f"hypothesis_{iteration}",
                    description=desc,
                    code=code,
                    score=None,
                    baseline_score=current_score,
                    status="failed",
                    error_message=err,
                    duration_sec=time.time() - iter_start,
                )
                continue

            # Оценка качества на 5-Fold CV
            X_cand_tr, y_cand_tr, X_cand_te, test_ids = self._extract_xy(cand_train, cand_test)
            train_res = self.trainer.train_cv(
                X_train=X_cand_tr,
                y_train=y_cand_tr,
                X_test=X_cand_te,
            )
            cand_score, _, cand_test_preds = self._postprocess_and_evaluate(train_res, y_cand_tr)
            iter_dur = time.time() - iter_start

            # Решение: Accept или Reject
            is_better = is_metric_better(
                new_score=cand_score,
                base_score=current_score,
                metric=self.metric,
                min_delta=self.min_improvement_delta,
            )

            if is_better:
                logger.info(f"🎉 ПРИНЯТО! Скор улучшился: {current_score:.5f} -> {cand_score:.5f}")
                current_score = cand_score
                current_train = cand_train
                current_test = cand_test
                self.best_score = cand_score
                self.best_train = cand_train
                self.best_test = cand_test
                self.latest_top_features = list(train_res.feature_importances.items())[:10]

                # Обновляем сабмит на диске
                is_numeric = not self.is_classification
                save_submission_safely(
                    ids=test_ids,
                    preds=cand_test_preds,
                    sample_sub_path=self.ctx.sample_sub_path,
                    output_path=sub_path,
                    is_numeric_target=is_numeric,
                )

                self.memory.log_experiment(
                    iteration=iteration,
                    name=f"hypothesis_{iteration}",
                    description=desc,
                    code=code,
                    score=cand_score,
                    baseline_score=self.baseline_score,
                    status="accepted",
                    duration_sec=iter_dur,
                )
            else:
                logger.info(f"❌ ОТКЛОНЕНО. Скор {cand_score:.5f} не превысил лучший {current_score:.5f}")
                self.memory.log_experiment(
                    iteration=iteration,
                    name=f"hypothesis_{iteration}",
                    description=desc,
                    code=code,
                    score=cand_score,
                    baseline_score=self.baseline_score,
                    status="rejected",
                    duration_sec=iter_dur,
                )

        return AutoResearchResult(
            baseline_score=self.baseline_score or 0.0,
            best_score=self.best_score or 0.0,
            best_train_df=self.best_train,
            best_test_df=self.best_test,
            accepted_records=self.memory.get_accepted_records(),
            memory=self.memory,
            sub_path=sub_path,
        )
