"""
Пакет ml_core: базовые модули машинного обучения ("руки" агента).

Включает:
- type_inference: предобработка строк и инференс типов (числа, даты, категории, текст).
- data_profiler: статистика, распределения и корреляции.
- splitter: стратегии разбиения выборки (Time, Group, Stratified, KFold).
- schemas: схемы данных (DataProfileReport, ModelSelectionDecision, SplitResult).
"""

try:
    from ml_core.dataset_loader import DatasetContext, load_dataset_context
except ImportError:
    DatasetContext = None
    load_dataset_context = None

try:
    from ml_core.schemas import DataProfileReport, EvalResult, ModelSelectionDecision, SplitResult
except ImportError:
    DataProfileReport = EvalResult = ModelSelectionDecision = SplitResult = None

try:
    from ml_core.evaluator import compute_metric, evaluate_model, evaluate_oof, is_metric_better
except ImportError:
    compute_metric = evaluate_model = evaluate_oof = is_metric_better = None

try:
    from ml_core.splitter import detect_split_strategy, get_cv_folds, split_data
except ImportError:
    detect_split_strategy = get_cv_folds = split_data = None

try:
    from ml_core.data_profiler import profile_data
    from ml_core.type_inference import infer_types
except ImportError:
    profile_data = None
    infer_types = None

try:
    from ml_core.postprocessor import apply_calibrated_predictions, optimize_class_weights, postprocess_regression
    from ml_core.submission_guard import SubmissionValidationError, save_submission_safely, validate_submission_data
except ImportError:
    apply_calibrated_predictions = optimize_class_weights = postprocess_regression = None
    SubmissionValidationError = save_submission_safely = validate_submission_data = None

__all__ = [
    "DatasetContext",
    "load_dataset_context",
    "infer_types",
    "profile_data",
    "split_data",
    "get_cv_folds",
    "detect_split_strategy",
    "DataProfileReport",
    "ModelSelectionDecision",
    "SplitResult",
    "EvalResult",
    "evaluate_model",
    "evaluate_oof",
    "is_metric_better",
    "compute_metric",
    "apply_calibrated_predictions",
    "optimize_class_weights",
    "postprocess_regression",
    "SubmissionValidationError",
    "save_submission_safely",
    "validate_submission_data",
]
