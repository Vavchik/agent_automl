"""
Модуль тяжелого ансамблирования и стэкинга (Tier 2 Upgrade: AutoGluon / LightAutoML).

Реализует:
1. Запуск мультимодельного стэкинга (AutoGluon TabularPredictor) на 8 CPU с параллелизацией.
2. Автоматический мэппинг метрик (Macro F1 -> f1_macro, RMSE -> root_mean_squared_error).
3. Извлечение OOF-скора стэка и генерация предсказаний на тестовой выборке.
4. Безопасный graceful fallback на случай отсутствия AutoGluon или нехватки RAM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Optional, Sequence

try:
    import pandas as pd
except ImportError:
    pd = None

from ml_core.evaluator import is_metric_better

logger = logging.getLogger("ml_core.automl_engine")


METRIC_MAPPING = {
    "rmse": "root_mean_squared_error",
    "root_mean_squared_error": "root_mean_squared_error",
    "mae": "mean_absolute_error",
    "macro_f1": "f1_macro",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


class AutoMLEnsembleEngine:
    """
    Движок глубокого табличного ансамблирования на базе AutoGluon.
    """

    def __init__(
        self,
        task_type: str = "multiclass_classification",
        metric: str = "macro_f1",
        time_limit_sec: int = 1200,
        num_cpus: int = 8,
    ):
        self.task_type = task_type
        self.metric = metric
        self.time_limit_sec = time_limit_sec
        self.num_cpus = num_cpus
        self.is_classification = "class" in task_type.lower()
        self.ag_metric = METRIC_MAPPING.get(metric.lower(), "f1_macro" if self.is_classification else "root_mean_squared_error")

    def is_available(self) -> bool:
        """Проверяет доступность AutoGluon в текущем окружении."""
        try:
            import autogluon.tabular  # noqa: F401
            return True
        except ImportError:
            return False

    def fit_and_predict(
        self,
        train_df: Any,
        test_df: Any,
        target_col: str,
        id_col: Optional[str] = None,
    ) -> tuple[Optional[float], Optional[list[Any]]]:
        """
        Обучает ансамбль моделей в AutoGluon и делает предсказания на тесте.
        Возвращает (ag_best_score, test_predictions) или (None, None) при сбое.
        """
        if pd is None or not isinstance(train_df, pd.DataFrame):
            logger.info("Pandas недоступен или входные данные не DataFrame. Пропуск AutoGluon.")
            return None, None

        if not self.is_available():
            logger.info("AutoGluon не установлен в окружении. Пропуск тяжелого ансамблирования.")
            return None, None

        from autogluon.tabular import TabularPredictor

        temp_dir = tempfile.mkdtemp(prefix="ag_ensemble_")
        try:
            logger.info(
                f"[AutoGluon] Запуск TabularPredictor: metric={self.ag_metric}, "
                f"time_limit={self.time_limit_sec}s, CPUs={self.num_cpus}"
            )

            # Подготавливаем датасеты без ID
            tr_data = train_df.drop(columns=[id_col]) if id_col and id_col in train_df.columns else train_df
            te_data = test_df.drop(columns=[id_col]) if id_col and id_col in test_df.columns else test_df

            predictor = TabularPredictor(
                label=target_col,
                eval_metric=self.ag_metric,
                path=temp_dir,
                problem_type="multiclass" if self.is_classification else "regression",
                verbosity=2,
            )

            # Обучаем стэк с включенным блендингом
            predictor.fit(
                train_data=tr_data,
                time_limit=self.time_limit_sec,
                presets="medium_quality",
                num_cpus=self.num_cpus,
                keep_only_best=True,
                save_space=True,
            )

            # Получаем лучший скор
            leaderboard = predictor.leaderboard(silent=True)
            best_score = None
            if not leaderboard.empty and "score_val" in leaderboard.columns:
                best_score = float(leaderboard["score_val"].iloc[0])
                # AutoGluon возвращает отрицательный RMSE (больше = лучше), инвертируем для RMSE
                if self.metric.lower() in ("rmse", "root_mean_squared_error", "mae") and best_score < 0:
                    best_score = -best_score

            # Предсказания на тесте
            preds = predictor.predict(te_data)
            test_preds = list(preds)

            logger.info(f"✓ AutoGluon успешно завершил обучение. Скор стэка: {best_score}")
            return best_score, test_preds

        except Exception as e:
            logger.error(f"Ошибка при работе AutoGluon: {e}", exc_info=True)
            return None, None
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
