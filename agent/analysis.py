"""Р4. Анализ ошибок лучшей модели по OOF — главная подсказка для следующей гипотезы.

Регрессия: в каких группах (категориях) сосредоточена ошибка и есть ли систематический сдвиг,
как модель ошибается на малых и больших значениях таргета.
Классификация: F1 по каждому классу и самые частые путаницы.
Результат — компактный dict для промпта.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from core.contracts import Dataset, ExperimentResult, TaskSpec
from ml.trainer import target_array


def _regression(ds: Dataset, spec: TaskSpec, result: ExperimentResult, top: int) -> dict:
    y, _ = target_array(ds, spec)
    resid = y - result.oof
    sq = resid ** 2
    total = sq.sum() or 1.0
    segments = []
    group_cols = list(ds.cat_cols)[:8] + ([spec.time_col] if spec.time_col in ds.X.columns else [])
    for col in group_cols:
        frame = pd.DataFrame({"g": ds.X[col].astype(str), "sq": sq, "r": resid})
        g = frame.groupby("g").agg(n=("sq", "size"), sq=("sq", "sum"), bias=("r", "mean"))
        g["error_share"] = g["sq"] / total
        g["rows_share"] = g["n"] / len(frame)
        g["excess"] = g["error_share"] - g["rows_share"]
        for value, row in g.sort_values("excess", ascending=False).head(2).iterrows():
            if row["excess"] > 0.02:
                segments.append({"column": col, "value": value, "rows_share": round(row["rows_share"], 3),
                                 "error_share": round(row["error_share"], 3),
                                 "rmse": round(float(np.sqrt(row["sq"] / row["n"])), 3),
                                 "bias_true_minus_pred": round(float(row["bias"]), 3)})
    segments = sorted(segments, key=lambda s: s["rows_share"] - s["error_share"])[:top]
    quintiles = pd.qcut(pd.Series(y).rank(method="first"), 5, labels=False)
    by_target = (pd.DataFrame({"q": quintiles, "r": resid, "sq": sq}).groupby("q")
                 .agg(bias=("r", "mean"), rmse=("sq", lambda s: float(np.sqrt(s.mean())))).round(3))
    return {
        "worst_segments": segments,
        "by_target_quintile": by_target.reset_index().to_dict(orient="records"),
        "largest_errors_share_top1pct": round(float(np.sort(sq)[-max(1, len(sq) // 100):].sum() / total), 3),
    }


def _classification(ds: Dataset, spec: TaskSpec, result: ExperimentResult, top: int) -> dict:
    y, classes = target_array(ds, spec)
    pred = result.oof.argmax(axis=1)
    f1 = f1_score(y, pred, average=None, labels=np.arange(len(classes)))
    support = np.bincount(y, minlength=len(classes))
    per_class = sorted(({"class": str(c), "f1": round(float(f), 3), "support": int(s)}
                        for c, f, s in zip(classes, f1, support)), key=lambda d: d["f1"])
    confusion = pd.crosstab(y, pred)
    pairs = []
    for t in confusion.index:
        for p in confusion.columns:
            if t != p and confusion.loc[t, p] > 0:
                pairs.append({"true": str(classes[t]), "predicted": str(classes[p]),
                              "count": int(confusion.loc[t, p]), "share_of_true": round(confusion.loc[t, p] / support[t], 3)})
    pairs = sorted(pairs, key=lambda d: -d["count"])[:top]
    return {"per_class_f1_worst_first": per_class, "top_confusions": pairs}


def error_analysis(ds: Dataset, spec: TaskSpec, result: ExperimentResult, top: int = 6) -> dict:
    try:
        if spec.task_type == "regression":
            return _regression(ds, spec, result, top)
        return _classification(ds, spec, result, top)
    except Exception as e:  # анализ — подсказка, а не обязательный шаг
        return {"error": str(e)[:200]}
