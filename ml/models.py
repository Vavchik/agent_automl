"""Р3. Фабрика моделей: одна функция обучает модель на фолде и возвращает предсказания.

Категориальные колонки приходят как pandas category с общими для train и test категориями:
LightGBM и XGBoost используют их напрямую, CatBoost получает строки.
Добавить модель = ветка в fit_predict + строка в DEFAULT_PARAMS.

Что улучшать (Р3): дефолтные параметры под 50-100 тыс. строк, линейная модель и kNN
для разнообразия ансамбля, DART, монотонные ограничения.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

DEFAULT_PARAMS: dict[str, dict] = {
    "lgbm": {"num_leaves": 63, "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
             "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"depth": 6, "l2_leaf_reg": 3.0},
    "xgb": {"max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0},
}


def _to_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype(str)
    return X


def fit_predict(name: str, task_type: str, X_tr, y_tr, X_va, y_va, X_te, cat_cols: list[str],
                params: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (предсказания на валидации, на test, важности признаков).
    Регрессия — значения, классификация — вероятности (n, k)."""
    p = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    regression = task_type == "regression"

    if name == "lgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
        model = cls(n_estimators=config.N_ESTIMATORS, learning_rate=config.LEARNING_RATE,
                    random_state=config.SEED, n_jobs=config.N_JOBS, verbose=-1, **p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
        importance = model.booster_.feature_importance("gain")
    elif name == "catboost":
        import catboost as cb
        cls = cb.CatBoostRegressor if regression else cb.CatBoostClassifier
        model = cls(iterations=config.N_ESTIMATORS, learning_rate=config.LEARNING_RATE,
                    random_seed=config.SEED, thread_count=config.N_JOBS, verbose=0,
                    early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, allow_writing_files=False, **p)
        X_tr, X_va, X_te = (_to_catboost(X, cat_cols) for X in (X_tr, X_va, X_te))
        model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va))
        importance = model.get_feature_importance()
    elif name == "xgb":
        import xgboost as xgb
        cls = xgb.XGBRegressor if regression else xgb.XGBClassifier
        model = cls(n_estimators=config.N_ESTIMATORS, learning_rate=config.LEARNING_RATE,
                    tree_method="hist", enable_categorical=True, random_state=config.SEED,
                    n_jobs=config.N_JOBS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, **p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        importance = model.feature_importances_
    else:
        raise ValueError(f"Неизвестная модель: {name}")

    if regression:
        return model.predict(X_va), model.predict(X_te), np.asarray(importance, dtype=float)
    return model.predict_proba(X_va), model.predict_proba(X_te), np.asarray(importance, dtype=float)
