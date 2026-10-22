"""Р5. Adversarial validation: насколько test отличается от train.

Учим модель отличать строки test от строк train. AUC около 0.5 — выборки похожи;
заметно выше — есть сдвиг, и случайный KFold будет завышать качество.
Самые важные признаки этой модели — те, что «сдвинулись»; их стоит показать агенту.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from core.contracts import Dataset
from runtime.logger import log


def adversarial_check(ds: Dataset, exclude: list[str] | None = None,
                      max_rows: int = 20_000) -> tuple[float, list[str]]:
    """exclude — колонки, которые отличаются по построению (например, период при time split)."""
    import lightgbm as lgb

    rng = np.random.RandomState(config.SEED)
    cols = [c for c in ds.X.columns if c not in (exclude or [])]
    tr = ds.X[cols].sample(min(len(ds.X), max_rows), random_state=rng)
    te = ds.X_test[cols].sample(min(len(ds.X_test), max_rows), random_state=rng)
    X = pd.concat([tr, te], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    oof = np.zeros(len(X))
    importance = np.zeros(X.shape[1])
    for tr_idx, va_idx in StratifiedKFold(3, shuffle=True, random_state=config.SEED).split(X, y):
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                   verbose=-1, n_jobs=config.N_JOBS)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        importance += model.feature_importances_
    auc = float(roc_auc_score(y, oof))
    top = [X.columns[i] for i in np.argsort(-importance)[:5]]
    log.info(f"Adversarial AUC train/test = {auc:.3f}; сильнее всего отличаются: {top}")
    return auc, top
