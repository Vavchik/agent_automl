"""Р3. AutoGluon best_quality на НАШИХ фолдах.

AutoGluon получает колонку с номером нашего фолда как groups, поэтому его бэггинг
валидируется ровно на тех же частях данных, что и все наши эксперименты (в т. ч. по году).
Его OOF-предсказания сравнимы с нашими и проходят то же правило принятия и тот же ансамбль.
Если AutoGluon не установлен, не уложился во время или упал — просто пропускаем.
"""
from __future__ import annotations

import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import config
from core.contracts import Dataset, ExperimentResult, Folds, TaskSpec
from ml.postprocess import evaluate
from ml.trainer import target_array
from runtime.logger import log

AG_METRICS = {
    "rmse": "root_mean_squared_error", "mse": "mean_squared_error", "mae": "mean_absolute_error",
    "r2": "r2", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "f1": "f1", "accuracy": "accuracy",
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "logloss": "log_loss",
}


def available() -> bool:
    try:
        import autogluon.tabular  # noqa: F401
        return True
    except Exception:
        return False


def run_autogluon(ds: Dataset, spec: TaskSpec, folds: Folds, time_limit: float) -> ExperimentResult | None:
    from autogluon.tabular import TabularPredictor

    start = time.time()
    y, classes = target_array(ds, spec)
    fold_id = np.zeros(len(ds.X), dtype=int)
    for k, (_, va) in enumerate(folds):
        fold_id[va] = k
    label, group = "__target__", "__fold__"
    train = ds.X.copy()
    train[label] = ds.y.to_numpy() if spec.is_classification else y
    train[group] = fold_id

    path = tempfile.mkdtemp(prefix="ag_")
    try:
        predictor = TabularPredictor(label=label, problem_type=spec.task_type, eval_metric=AG_METRICS.get(spec.metric),
                                     groups=group, path=path, verbosity=1)
        predictor.fit(train, presets="best_quality", time_limit=int(time_limit), dynamic_stacking=False,
                      num_cpus=config.N_JOBS, excluded_model_types=["KNN"])
        if spec.is_classification:
            oof = predictor.predict_proba_oof()[list(classes)].to_numpy()
            test_pred = predictor.predict_proba(ds.X_test)[list(classes)].to_numpy()
        else:
            oof = predictor.predict_oof().to_numpy(dtype=float)
            test_pred = predictor.predict(ds.X_test).to_numpy(dtype=float)
    except Exception as e:
        log.warning(f"AutoGluon не удался: {e}")
        return None
    finally:
        shutil.rmtree(path, ignore_errors=True)

    result = ExperimentResult(
        name="autogluon",
        score=evaluate(spec, y, oof),
        fold_scores=[evaluate(spec, y[va], oof[va]) for _, va in folds],
        oof=oof, test_pred=test_pred, classes=classes, seconds=time.time() - start,
    )
    log.info(f"[autogluon] {spec.metric} = {result.score:.5f} за {result.seconds:.0f} с")
    return result
