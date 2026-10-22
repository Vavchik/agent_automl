"""Р2. LLM читает task.md и дополняет TaskSpec тем, что эвристики не видят.

Политика сейчас простая: метрику и тип задачи берём из эвристик (они проверены на данных),
а от LLM — колонку времени/группы и короткие заметки о задаче для промптов агента.
Если LLM уверенно называет другую метрику из реестра, логируем расхождение — решение за Р2.
"""
from __future__ import annotations

import pandas as pd

from agent.llm import llm
from core.contracts import TaskSpec
from runtime.logger import log
from task.metrics import METRICS

SYSTEM = (
    "Ты ML-инженер. Прочитай описание задачи и верни ТОЛЬКО JSON без пояснений с полями: "
    '{"metric": один из ' + ", ".join(METRICS) + ', '
    '"time_col": имя колонки периода/даты или null, '
    '"group_col": имя колонки сущности (клиент, здание), если строки одной сущности повторяются, или null, '
    '"notes": 3-5 коротких фактов о данных и предметной области, важных для признаков}'
)


def refine_with_llm(spec: TaskSpec, train: pd.DataFrame) -> TaskSpec:
    if not llm.enabled:
        return spec
    user = f"Описание задачи:\n{spec.text[:6000]}\n\nКолонки train: {list(train.columns)[:200]}"
    answer = llm.ask_json(SYSTEM, user, max_tokens=800)
    if not answer:
        return spec
    llm_metric = str(answer.get("metric") or "").lower()
    if llm_metric in METRICS and llm_metric != spec.metric:
        log.warning(f"LLM считает метрикой '{llm_metric}', эвристики — '{spec.metric}'. Оставляем эвристику.")
    for field in ("time_col", "group_col"):
        value = answer.get(field)
        if isinstance(value, str) and value in train.columns and value not in (spec.target_col, spec.id_col):
            setattr(spec, field, value)
    notes = answer.get("notes")
    if notes:
        spec.notes = notes if isinstance(notes, str) else "; ".join(map(str, notes))
    log.info(f"LLM-разбор задачи: time_col={spec.time_col}, group_col={spec.group_col}")
    return spec
