"""Р3. Ансамбль из уже посчитанных экспериментов (жадный отбор по OOF, Caruana et al.).

Все эксперименты считались на одних фолдах, поэтому их OOF можно смешивать.
Результат проходит то же правило принятия, что и любая гипотеза.

Что улучшать (Р3): стекинг (логрег/ridge поверх OOF), AutoGluon на тех же фолдах.
"""
from __future__ import annotations

import time

import numpy as np

from core.contracts import ExperimentResult, Folds, TaskSpec
from ml.postprocess import evaluate
from task.metrics import improvement


def greedy_blend(results: list[ExperimentResult], spec: TaskSpec, y: np.ndarray, folds: Folds,
                 max_steps: int = 15) -> ExperimentResult | None:
    pool = [r for r in results if r.oof is not None]
    if len(pool) < 2:
        return None
    start = time.time()
    best_idx = max(range(len(pool)), key=lambda i: improvement(spec.metric, pool[i].score, pool[0].score))
    chosen = [best_idx]
    current = pool[best_idx].oof.copy()
    current_score = pool[best_idx].score
    for _ in range(max_steps):
        trials = []
        for i, r in enumerate(pool):
            blended = (current * len(chosen) + r.oof) / (len(chosen) + 1)
            trials.append((evaluate(spec, y, blended), i, blended))
        s, i, blended = max(trials, key=lambda t: improvement(spec.metric, t[0], current_score))
        if improvement(spec.metric, s, current_score) <= 0:
            break
        chosen.append(i)
        current, current_score = blended, s
    if len(set(chosen)) < 2:
        return None
    weights = np.bincount(chosen, minlength=len(pool)) / len(chosen)
    test_pred = sum(w * r.test_pred for w, r in zip(weights, pool) if w > 0)
    names = [f"{pool[i].name}×{weights[i]:.2f}" for i in np.nonzero(weights)[0]]
    return ExperimentResult(
        name="blend(" + ", ".join(names) + ")",
        score=current_score,
        fold_scores=[evaluate(spec, y[va], current[va]) for _, va in folds],
        oof=current,
        test_pred=test_pred,
        classes=pool[0].classes,
        seconds=time.time() - start,
    )
