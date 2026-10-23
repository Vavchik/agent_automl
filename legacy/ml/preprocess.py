"""Р3. Преобразования таргета для обучения и OOF-кодирование признаков.

Признаки готовит task/dtypes.py (общие категории для train и test),
здесь — то, что зависит от таргета (TargetTransform и leakage-free OOF Target Encoding).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_labels(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Метки -> коды 0..k-1 (классы отсортированы). Возвращает (коды, классы)."""
    codes, classes = pd.factorize(y, sort=True)
    return codes.astype(int), np.asarray(classes)


class TargetTransform:
    """None — без изменений; "log1p" — для неотрицательного таргета с длинным хвостом."""

    def __init__(self, kind: str | None, y: np.ndarray):
        self.kind = kind if (kind == "log1p" and np.nanmin(y) > -1) else None

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y) if self.kind == "log1p" else y

    def inverse(self, p: np.ndarray) -> np.ndarray:
        return np.expm1(p) if self.kind == "log1p" else p


def compute_oof_target_encoding(
    X_tr: pd.DataFrame,
    y_tr: pd.Series | np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_type: str,
    smoothing: float = 10.0,
    max_cats: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сглаженный Out-Of-Fold Target Encoding без утечек данных для категорий."""
    eligible_cats = [c for c in cat_cols if c in X_tr.columns and X_tr[c].nunique() >= 2][:max_cats]
    if not eligible_cats:
        return pd.DataFrame(index=X_tr.index), pd.DataFrame(index=X_te.index)

    y_arr = np.asarray(y_tr)
    n_train = len(X_tr)
    new_tr_cols: dict[str, np.ndarray] = {}
    new_te_cols: dict[str, np.ndarray] = {}

    if task_type in ("binary", "regression"):
        y_float = y_arr.astype(float)
        global_prior = float(np.nanmean(y_float))
        for col in eligible_cats:
            s_tr = X_tr[col].astype(str)
            s_te = X_te[col].astype(str)
            oof_col = np.full(n_train, global_prior, dtype=float)
            for tr_idx, va_idx in folds:
                tr_series = s_tr.iloc[tr_idx]
                tr_y = y_float[tr_idx]
                prior = float(np.nanmean(tr_y))
                df_fold = pd.DataFrame({"cat": tr_series, "y": tr_y})
                grp = df_fold.groupby("cat")["y"].agg(["count", "mean"])
                smooth = (grp["count"] * grp["mean"] + smoothing * prior) / (grp["count"] + smoothing)
                va_series = s_tr.iloc[va_idx]
                oof_col[va_idx] = va_series.map(smooth).fillna(prior).to_numpy()
            new_tr_cols[f"{col}__target_enc"] = oof_col

            df_full = pd.DataFrame({"cat": s_tr, "y": y_float})
            grp_full = df_full.groupby("cat")["y"].agg(["count", "mean"])
            smooth_full = (grp_full["count"] * grp_full["mean"] + smoothing * global_prior) / (grp_full["count"] + smoothing)
            new_te_cols[f"{col}__target_enc"] = s_te.map(smooth_full).fillna(global_prior).to_numpy()

    elif task_type == "multiclass":
        unique_classes = np.unique(y_arr)[:10]
        for col in eligible_cats:
            s_tr = X_tr[col].astype(str)
            s_te = X_te[col].astype(str)
            for cl in unique_classes:
                y_bin = (y_arr == cl).astype(float)
                global_prior = float(np.nanmean(y_bin))
                oof_col = np.full(n_train, global_prior, dtype=float)
                for tr_idx, va_idx in folds:
                    tr_series = s_tr.iloc[tr_idx]
                    tr_y = y_bin[tr_idx]
                    prior = float(np.nanmean(tr_y))
                    df_fold = pd.DataFrame({"cat": tr_series, "y": tr_y})
                    grp = df_fold.groupby("cat")["y"].agg(["count", "mean"])
                    smooth = (grp["count"] * grp["mean"] + smoothing * prior) / (grp["count"] + smoothing)
                    va_series = s_tr.iloc[va_idx]
                    oof_col[va_idx] = va_series.map(smooth).fillna(prior).to_numpy()
                new_tr_cols[f"{col}__te_cl_{cl}"] = oof_col

                df_full = pd.DataFrame({"cat": s_tr, "y": y_bin})
                grp_full = df_full.groupby("cat")["y"].agg(["count", "mean"])
                smooth_full = (grp_full["count"] * grp_full["mean"] + smoothing * global_prior) / (grp_full["count"] + smoothing)
                new_te_cols[f"{col}__te_cl_{cl}"] = s_te.map(smooth_full).fillna(global_prior).to_numpy()

    return pd.DataFrame(new_tr_cols, index=X_tr.index), pd.DataFrame(new_te_cols, index=X_te.index)
