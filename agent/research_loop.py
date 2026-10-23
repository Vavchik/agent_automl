"""Р4. AutoResearch как поиск по дереву решений.

Итерация: выбрать узел (tree.select_parent) -> выбрать действие (planner) -> получить кандидата
(actions, с починкой кода по traceback) -> обучить на тех же фолдах (ml) -> оценить (validation):
  - прирост к родителю устойчив -> узел «удачный», его можно расширять и сливать с другими ветками;
  - лучше лучшего во всём дереве -> новый лучший узел, submission.csv сразу обновляется.
Остановка платформой в любой момент оставляет лучший найденный ответ.
"""
from __future__ import annotations

import time
from typing import Callable

import config
from agent.actions import ACTIONS, ResearchState
from agent.memory import Record
from agent.planner import next_action
from agent.tree import Node
from ml.trainer import train_cv
from runtime.logger import log
from task.metrics import improvement
from validation.acceptance import should_accept


def run_research(state: ResearchState, on_new_best: Callable[[Node], None], reserve_sec: float = 0.0) -> None:
    tree = state.tree
    last_duration = tree.best.result.seconds
    for iteration in range(1, config.MAX_RESEARCH_ITERS + 1):
        if not state.clock.has(1.5 * last_duration + reserve_sec + 30):
            log.info("Время на исследование закончилось")
            break
        parent = tree.select_parent()
        if parent is None:
            log.info("В дереве не осталось узлов для расширения")
            break
        name = next_action(state, parent)
        if name is None:
            parent.exhausted = True
            continue
        log.info(f"--- Итерация {iteration}: {name} от узла #{parent.id} ({parent.result.score:.5f}), "
                 f"осталось {state.clock.remaining() / 60:.0f} мин ---")
        start = time.time()
        node = Node(id=tree.new_id(), parent_id=parent.id, action=name, hypothesis="", result=None, ds=None,
                    cfg=None, depth=parent.depth + 1, lineage=parent.lineage | {name})
        try:
            candidate = ACTIONS[name].run(state, parent)
            if candidate is None:
                parent.skipped.add(name)
                log.info(f"Действию {name} нечего предложить для #{parent.id}")
                continue
            if candidate.added_cols and any(sorted(n.added_cols) == sorted(candidate.added_cols)
                                            for n in tree.nodes.values() if n.parent_id == parent.id):
                log.info(f"Повтор уже проверенных признаков для #{parent.id} — пропускаем без обучения")
                continue
            node.hypothesis, node.code, node.added_cols = candidate.hypothesis, candidate.code, candidate.added_cols
            node.lineage = node.lineage | ({candidate.tag} if candidate.tag else set())
            result = train_cv(candidate.ds, state.spec, state.folds, candidate.cfg, name=f"#{node.id}:{name}")
        except Exception as e:
            node.error = str(e)[:1500]
            tree.add(node)
            log.warning(f"Узел #{node.id} ({name}) упал: {node.error[:300]}")
            state.memory.log(Record(iteration, name, node.hypothesis, None, parent.result.score, False, "",
                                    time.time() - start, error=node.error, code=node.code,
                                    node_id=node.id, parent_id=parent.id))
            continue

        node.result, node.ds, node.cfg = result, candidate.ds, candidate.cfg
        node.gain = improvement(state.spec.metric, result.score, parent.result.score)
        node.improving, reason = should_accept(result, parent.result, state.spec.metric)
        tree.add(node)
        state.results.append(result)
        is_best, best_reason = should_accept(result, tree.best.result, state.spec.metric)
        state.memory.log(Record(iteration, name, node.hypothesis, result.score, parent.result.score, node.improving,
                                reason, time.time() - start, code=node.code, node_id=node.id, parent_id=parent.id))
        log.info(f"{'УДАЧА' if node.improving else 'без прироста'} #{node.id}: {node.hypothesis[:150]} | {reason}")
        if is_best:
            tree.best_id = node.id
            log.info(f"Новый лучший узел #{node.id}: {result.score:.5f} ({best_reason})")
            on_new_best(node)
        tree.prune()
        last_duration = time.time() - start
