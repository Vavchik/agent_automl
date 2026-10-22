"""Р4. Цикл AutoResearch: гипотеза -> эксперимент -> оценка -> следующая итерация.

Каждая принятая гипотеза сразу обновляет submission.csv (через колбэк on_new_best),
поэтому остановка платформой в любой момент оставляет лучший найденный ответ.
"""
from __future__ import annotations

import time
from typing import Callable

import config
from agent.actions import ACTIONS, ResearchState
from agent.memory import Record
from agent.planner import next_action
from core.contracts import ExperimentResult
from ml.trainer import train_cv
from runtime.logger import log
from validation.acceptance import should_accept


def run_research(state: ResearchState, on_new_best: Callable[[ExperimentResult, ResearchState], None]) -> None:
    last_duration = state.best.seconds
    for iteration in range(1, config.MAX_RESEARCH_ITERS + 1):
        final_reserve = config.FINAL_TIME_FACTOR * state.best.seconds   # время на финальный бленд
        if not state.clock.has(1.5 * last_duration + final_reserve + 30):
            log.info("Время на исследование закончилось")
            break
        name = next_action(state)
        if name is None:
            log.info("Доступных действий больше нет")
            break
        log.info(f"--- Итерация {iteration}: {name} (осталось {state.clock.remaining() / 60:.0f} мин) ---")
        start = time.time()
        best_before = state.best.score
        try:
            candidate = ACTIONS[name].run(state)
            if candidate is None:
                state.memory.log(Record(iteration, name, "нечего пробовать", None, best_before, False, "",
                                        time.time() - start))
                continue
            result = train_cv(candidate.ds, state.spec, state.folds, candidate.cfg, name=f"{iteration}:{name}")
        except Exception as e:
            log.warning(f"Итерация {iteration} ({name}) упала: {e}")
            state.memory.log(Record(iteration, name, "", None, best_before, False, "", time.time() - start,
                                    error=str(e)[:1500]))
            continue

        accepted, reason = should_accept(result, state.best, state.spec.metric)
        state.results.append(result)
        state.memory.log(Record(iteration, name, candidate.hypothesis, result.score, best_before, accepted,
                                reason, time.time() - start, code=candidate.code))
        log.info(f"{'ПРИНЯТО' if accepted else 'отклонено'}: {candidate.hypothesis[:150]} | {reason}")
        if accepted:
            state.ds, state.cfg, state.best = candidate.ds, candidate.cfg, result
            on_new_best(result, state)
        last_duration = time.time() - start
