"""
Главный координатор (оркестратор) AutoML-пайплайна.

Связывает стадии через LangGraph граф или пошаговый API:
1. Очистка типов и сбор профиля данных (жесткий код: `ml_core.type_inference`, `ml_core.data_profiler`).
2. Мета-анализ и выбор моделей (АГЕНТ: `agent.meta_analyst` / LLM Qwen).
3. Разбиение данных на выборки (жесткий код: `ml_core.splitter`).
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from agent.graph import app as pipeline_graph, make_initial_state, PipelineState
from agent.meta_analyst import select_model_and_metric
from ml_core.data_profiler import profile_data
from ml_core.evaluator import evaluate_model
from ml_core.schemas import DataProfileReport, EvalResult, ModelSelectionDecision, SplitResult
from ml_core.splitter import split_data
from ml_core.type_inference import infer_types


class AutoMLOrchestrator:
    """
    Класс оркестрации пайплайна AutoML:
    Управляет сквозным выполнением графа с 3-уровневым каскадом (Tier 1/2/3).
    """

    def __init__(
        self,
        target: str,
        time_col: Optional[str] = None,
        group_col: Optional[str] = None,
        target_metric_threshold: float = 0.0,
    ):
        self.target = target
        self.time_col = time_col
        self.group_col = group_col
        self.target_metric_threshold = target_metric_threshold

        self.clean_df: Optional[pd.DataFrame] = None
        self.column_types: dict[str, str] = {}
        self.report: Optional[DataProfileReport] = None
        self.decision: Optional[ModelSelectionDecision] = None
        self.splits: Optional[SplitResult] = None
        self.trained_model: Optional[Any] = None
        self.eval_result: Optional[EvalResult] = None
        self.tier: int = 1

    def run_pipeline(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        target_metric_threshold: Optional[float] = None,
    ) -> PipelineState:
        """
        Запускает полный каскадный пайплайн через LangGraph:
        profiling -> meta_analysis -> splitting -> evaluate -> (automl -> agent_coder).
        """
        threshold = (
            target_metric_threshold
            if target_metric_threshold is not None
            else self.target_metric_threshold
        )
        initial_state = make_initial_state(
            df=df,
            target=self.target,
            time_col=self.time_col,
            group_col=self.group_col,
            test_size=test_size,
            val_size=val_size,
            target_metric_threshold=threshold,
        )
        final_state = pipeline_graph.invoke(initial_state)

        # Сохраняем результаты в поля оркестратора
        self.clean_df = final_state.get("clean_df")
        self.column_types = final_state.get("column_types", {})
        self.report = final_state.get("report")
        self.decision = final_state.get("decision")
        self.splits = final_state.get("splits")
        self.trained_model = final_state.get("trained_model")
        self.eval_result = final_state.get("eval_result")
        self.tier = final_state.get("tier", 1)

        return final_state

    # ----------------------------------------------------------------------- #
    # Пошаговые методы (для детального контроля стадий)
    # ----------------------------------------------------------------------- #

    def run_stage_profile(self, df: pd.DataFrame) -> DataProfileReport:
        """Стадия 1 (Жесткий код): Очистка мусора, определение типов и сбор профиля."""
        self.clean_df, self.column_types = infer_types(df)
        self.report = profile_data(self.clean_df, self.column_types, target=self.target)
        return self.report

    def run_stage_meta_analysis(self) -> ModelSelectionDecision:
        """Стадия 2 (АГЕНТ): Выбор архитектур, лосса и метрики через LLM Qwen."""
        if self.report is None:
            raise ValueError("Сначала запустите run_stage_profile()")
        self.decision = select_model_and_metric(self.report)
        return self.decision

    def run_stage_split(self, test_size: float = 0.2, val_size: float = 0.1) -> SplitResult:
        """Стадия 3 (Жесткий код): Детерминированное разбиение выборки."""
        if self.clean_df is None:
            raise ValueError("Сначала запустите run_stage_profile()")

        strategy = self.decision.split_strategy if self.decision else None
        self.splits = split_data(
            self.clean_df,
            target=self.target,
            strategy=strategy,
            test_size=test_size,
            val_size=val_size,
            time_col=self.time_col,
            group_col=self.group_col,
        )
        return self.splits

    def run_stage_evaluate(
        self,
        model: Optional[Any] = None,
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> EvalResult:
        """Стадия 6 (Жесткий код): Оценка модели, бейзлайн и проверка оверфита."""
        target_model = model or self.trained_model
        if target_model is None:
            raise ValueError("Необходимо передать обученную модель или сначала обучить ее")
        if self.splits is None:
            raise ValueError("Сначала запустите run_stage_split()")

        target_metric = metric or (self.decision.metric if self.decision else "roc_auc")
        target_thresh = threshold if threshold is not None else self.target_metric_threshold

        X_train = self.splits.train.drop(columns=[self.target])
        y_train = self.splits.train[self.target]

        val_df = (
            self.splits.val
            if (self.splits.val is not None and len(self.splits.val) > 0)
            else self.splits.test
        )
        X_val = val_df.drop(columns=[self.target])
        y_val = val_df[self.target]

        self.eval_result = evaluate_model(
            model=target_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metric=target_metric,
            threshold=target_thresh,
        )
        self.trained_model = target_model
        return self.eval_result