"""Р5. Правило принятия гипотезы: прирост должен быть больше шума.

Все эксперименты считаются на одних и тех же фолдах, поэтому сравниваем попарно:
d_i = улучшение на фолде i. Принимаем, если mean(d) > K * std(d) / sqrt(n),
то есть прирост устойчив, а не случайная удача на одном фолде.
K задаётся в config.ACCEPT_K (1.0 — умеренно строго).

Что улучшать (Р5): повтор на другом сиде для спорных случаев, отложенная выборка,
которую агент видит один раз в конце.
"""
from __future__ import annotations

import numpy as np

import config
from core.contracts import ExperimentResult
from task.metrics import METRICS, improvement


def should_accept(new: ExperimentResult, best: ExperimentResult, metric: str,
                  k: float = config.ACCEPT_K) -> tuple[bool, str]:
    sign = 1.0 if METRICS[metric].higher_is_better else -1.0
    diffs = sign * (np.asarray(new.fold_scores) - np.asarray(best.fold_scores))
    gain = improvement(metric, new.score, best.score)
    noise = k * diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    ok = gain > 0 and diffs.mean() > noise
    reason = f"прирост {gain:+.5f}, по фолдам {diffs.mean():+.5f} ± {noise:.5f}"
    return bool(ok), reason
