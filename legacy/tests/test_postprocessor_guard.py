"""
Тесты для модуля калибровки (postprocessor) и валидатора сабмитов (submission_guard).
"""

from pathlib import Path
import numpy as np
import pytest

from ml_core.postprocessor import (
    apply_calibrated_predictions,
    optimize_class_weights,
    postprocess_regression,
)
from ml_core.submission_guard import (
    SubmissionValidationError,
    save_submission_safely,
    validate_submission_data,
)


def test_submission_validation_success():
    ids = [1, 2, 3]
    preds = [10.5, 20.1, 30.0]
    sample_ids = [1, 2, 3]
    is_valid, msg = validate_submission_data(ids, preds, sample_ids, is_numeric_target=True)
    assert is_valid is True
    assert msg == "OK"


def test_submission_validation_nan_failure():
    ids = [1, 2, 3]
    preds = [10.5, float("nan"), 30.0]
    sample_ids = [1, 2, 3]
    is_valid, msg = validate_submission_data(ids, preds, sample_ids, is_numeric_target=True)
    assert is_valid is False
    assert "NaN" in msg


def test_submission_validation_id_mismatch():
    ids = [1, 3, 2]
    preds = ["Normal", "Generic", "Exploits"]
    sample_ids = [1, 2, 3]
    is_valid, msg = validate_submission_data(ids, preds, sample_ids)
    assert is_valid is False
    assert "Несовпадение ID" in msg


def test_save_submission_safely_atomic(tmp_path: Path):
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,attack_cat\n101,Normal\n102,Generic\n103,DoS\n", encoding="utf-8")

    out_path = tmp_path / "submission.csv"
    res_path = save_submission_safely(
        ids=[101, 102, 103],
        preds=["Normal", "Generic", "DoS"],
        sample_sub_path=sample_path,
        output_path=out_path,
    )
    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "id,attack_cat"
    assert content[1] == "101,Normal"
    assert content[3] == "103,DoS"


def test_postprocess_regression_clipping():
    raw_preds = [-5.0, 10.0, 25.5, 999.0]
    processed = postprocess_regression(raw_preds, lower_bound=0.0, upper_bound=500.0)
    assert processed[0] == 0.0  # Отрицательное скорректировано до 0
    assert processed[1] == 10.0
    assert processed[3] == 500.0  # Верхний предел


def test_optimize_class_weights_imbalanced():
    classes = ["Rare", "Common"]
    # Создаем вероятности, где редкий класс слегка занижен
    y_true = ["Rare", "Rare", "Common", "Common", "Common"]
    # Несбалансированные вероятности
    oof_probs = np.array([
        [0.45, 0.55],
        [0.40, 0.60],
        [0.10, 0.90],
        [0.05, 0.95],
        [0.20, 0.80],
    ])
    weights = optimize_class_weights(oof_probs, y_true, classes, metric="macro_f1")
    # Вес для редкого класса должен возрасти
    assert weights[0] >= 1.0

    calibrated_preds = apply_calibrated_predictions(oof_probs, classes, weights)
    assert len(calibrated_preds) == len(y_true)
