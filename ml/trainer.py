"""Р3. Кросс-валидация на заданных фолдах: OOF-предсказания, среднее по фолдам для test.

Несколько моделей в ModelConfig.models усредняются на каждом фолде (простой бленд).
Скор считается по OOF на тех же фолдах для всех экспериментов — поэтому они сравнимы.
"""
from __future__ import annotations

import time

import numpy as np

from core.contracts import Dataset, ExperimentResult, Folds, ModelConfig, TaskSpec
from ml.models import fit_predict
from ml.postprocess import evaluate
from ml.preprocess import TargetTransform, encode_labels
from runtime.logger import log


def target_array(ds: Dataset, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray | None]:
    """Таргет в виде для метрик: значения (регрессия) или коды классов + классы."""
    if spec.is_classification:
        return encode_labels(ds.y)
    return ds.y.to_numpy(dtype=float), None


def train_cv(ds: Dataset, spec: TaskSpec, folds: Folds, cfg: ModelConfig, name: str = "") -> ExperimentResult:
    start = time.time()
    y, classes = target_array(ds, spec)
    n, m = len(ds.X), len(ds.X_test)
    k = len(classes) if classes is not None else 0
    oof = np.zeros((n, k)) if k else np.zeros(n)
    test_pred = np.zeros((m, k)) if k else np.zeros(m)
    importance = np.zeros(ds.X.shape[1])
    transform = TargetTransform(cfg.target_transform, y) if not k else None
    y_fit = transform.forward(y) if transform else y
    fold_scores = []

    for tr, va in folds:
        va_preds, te_preds = [], []
        for model_name in cfg.models:
            p_va, p_te, imp = fit_predict(model_name, spec.task_type, ds.X.iloc[tr], y_fit[tr],
                                          ds.X.iloc[va], y_fit[va], ds.X_test, ds.cat_cols,
                                          cfg.params.get(model_name))
            va_preds.append(p_va)
            te_preds.append(p_te)
            if imp.sum() > 0:
                importance += imp / imp.sum()
        p_va, p_te = np.mean(va_preds, axis=0), np.mean(te_preds, axis=0)
        if transform:
            p_va, p_te = transform.inverse(p_va), transform.inverse(p_te)
        oof[va] = p_va
        test_pred += p_te / len(folds)
        fold_scores.append(evaluate(spec, y[va], p_va))

    result = ExperimentResult(
        name=name,
        score=evaluate(spec, y, oof),
        fold_scores=fold_scores,
        oof=oof,
        test_pred=test_pred,
        classes=classes,
        seconds=time.time() - start,
        feature_importance=dict(sorted(zip(ds.X.columns, importance), key=lambda t: -t[1])),
    )
    log.info(f"[{name}] {spec.metric} = {result.score:.5f} (фолды: {np.round(fold_scores, 4).tolist()}), "
             f"{result.seconds:.0f} с")
    return result
