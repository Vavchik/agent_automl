"""Р4. Цикл AutoResearch на основе Solution Tree (дерева решений с бэктрекингом).

- Каждая итерация порождает узел в SolutionTree.
- При успехе узел принимается как лучший и обновляет submission.csv.
- При отклонении или падении агент выполняет Backtracking к лучшему узлу дерева,
  не накапливая неудачные мутации признаков.
- Топовые узлы дерева затем используются для финального блендинга.
"""
from __future__ import annotations

import time
from typing import Callable

import config
from agent.actions import ACTIONS, ResearchState
from agent.memory import Record
from agent.planner import next_action
from agent.solution_tree import SolutionTree
from core.contracts import ExperimentResult
from ml.trainer import train_cv
from runtime.logger import log
from validation.acceptance import should_accept


def run_research(state: ResearchState, on_new_best: Callable[[ExperimentResult, ResearchState], None]) -> None:
    if state.tree is None:
        state.tree = SolutionTree(state.ds.copy(), state.cfg.copy(), state.best)

    last_duration = state.best.seconds
    for iteration in range(1, config.MAX_RESEARCH_ITERS + 1):
        final_reserve = config.FINAL_TIME_FACTOR * state.best.seconds
        if not state.clock.has(1.5 * last_duration + final_reserve + 30):
            log.info("Время на исследование подошло к лимиту, переход к финализации")
            break

        # Backtracking / выбор узла для ветвления
        active_node = state.tree.get_exploration_node()
        state.ds = active_node.ds.copy()
        state.cfg = active_node.cfg.copy()
        if active_node.result is not None:
            state.best = active_node.result

        name = next_action(state)
        if name is None:
            log.info("Доступных действий больше нет")
            break

        log.info(f"--- Итерация {iteration}: {name} (от узла #{active_node.id}, осталось {state.clock.remaining() / 60:.0f} мин) ---")
        start = time.time()
        best_before = state.best.score
        candidate = None
        tree_node = None

        try:
            candidate = ACTIONS[name].run(state)
            if candidate is None:
                state.memory.log(Record(iteration, name, "нечего пробовать", None, best_before, False, "",
                                        time.time() - start))
                continue

            tree_node = state.tree.add_candidate(
                parent_id=active_node.id,
                iteration=iteration,
                action=name,
                hypothesis=candidate.hypothesis,
                ds=candidate.ds,
                cfg=candidate.cfg,
                code=candidate.code,
            )

            result = train_cv(candidate.ds, state.spec, state.folds, candidate.cfg, name=f"{iteration}:{name}")
        except Exception as e:
            err_msg = str(e)[:1500]
            log.warning(f"Итерация {iteration} ({name}) упала: {e}")
            if tree_node is not None:
                state.tree.record_failure(tree_node.id, err_msg)
            state.memory.log(Record(iteration, name, "", None, best_before, False, "", time.time() - start,
                                    error=err_msg))
            continue

        accepted, reason = should_accept(result, state.best, state.spec.metric)
        state.results.append(result)
        state.tree.record_result(tree_node.id, result, accepted, reason)
        state.memory.log(Record(iteration, name, candidate.hypothesis, result.score, best_before, accepted,
                                reason, time.time() - start, code=candidate.code))

        log.info(f"{'ПРИНЯТО' if accepted else 'отклонено'}: {candidate.hypothesis[:150]} | {reason}")
        if accepted:
            state.ds, state.cfg, state.best = candidate.ds, candidate.cfg, result
            on_new_best(result, state)
        last_duration = time.time() - start
