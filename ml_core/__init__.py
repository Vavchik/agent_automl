"""
Пакет ml_core: базовые модули машинного обучения ("руки" агента).

Включает:
- type_inference: предобработка строк и инференс типов (числа, даты, категории, текст).
- data_profiler: статистика, распределения и корреляции.
- splitter: стратегии разбиения выборки (Time, Group, Stratified, KFold).
- schemas: схемы данных (DataProfileReport, ModelSelectionDecision, SplitResult).
"""

from ml_core.data_profiler import profile_data
from ml_core.evaluator import compute_metric, evaluate_model
from ml_core.schemas import DataProfileReport, EvalResult, ModelSelectionDecision, SplitResult
from ml_core.splitter import detect_split_strategy, split_data
from ml_core.type_inference import infer_types

__all__ = [
    "infer_types",
    "profile_data",
    "split_data",
    "detect_split_strategy",
    "DataProfileReport",
    "ModelSelectionDecision",
    "SplitResult",
    "EvalResult",
    "evaluate_model",
    "compute_metric",
]
