"""Р4. Выбор действия для узла, который дерево решило расширить.

Если LLM доступна, она выбирает действие по анализу ошибок и истории дерева.
Иначе — простые правила: сначала дешёвые одноразовые проверки (log-таргет, веса классов),
затем слияние удачных веток, признаки, а тюнинг — один раз, когда решение уже созрело.

Что улучшать (Р4): учитывать стоимость действия по времени и ожидаемый прирост.
"""
from __future__ import annotations

import json

from agent.actions import ACTIONS, ResearchState, parent_tried
from agent.llm import llm
from agent.tree import Node
from runtime.logger import log

PLANNER_SYSTEM = """Ты руководишь исследованием ML-модели для табличной задачи. Решения хранятся деревом.
Для выбранного решения выбери ОДНО следующее действие из списка доступных.
Ориентируйся на анализ ошибок: если ошибка сосредоточена в сегментах или классах, нужны признаки;
если есть удачные соседние ветки, их стоит объединить; тюнинг полезен, когда признаки уже сильные.
Верни только JSON: {"action": "<имя>", "why": "<одно предложение>"}"""


def available_actions(state: ResearchState, parent: Node) -> list[str]:
    tried = parent_tried(parent, state)
    failed_once = {n.action for n in state.tree.nodes.values() if n.parent_id is not None and not n.improving}
    names = []
    for a in ACTIONS.values():
        if a.name in parent.skipped:
            continue
        # одноразовое действие: не повторяем на пути и не повторяем, если оно уже не помогло в другой ветке
        if not a.repeatable and (a.name in parent.lineage or a.name in tried or a.name in failed_once):
            continue
        if a.can_run(state, parent):
            names.append(a.name)
    return names


def _tuned_anywhere(state: ResearchState) -> bool:
    return any(n.action == "tune_lgbm" for n in state.tree.nodes.values())


def next_action(state: ResearchState, parent: Node) -> str | None:
    options = available_actions(state, parent)
    if not options:
        return None
    if llm.available() and len(state.tree.nodes) >= 3:
        menu = "\n".join(f"- {name}: {ACTIONS[name].description}" for name in options)
        user = (f"Метрика {state.spec.metric}. Расширяем решение #{parent.id} (скор {parent.result.score:.5f}), "
                f"осталось {state.clock.remaining() / 60:.0f} мин.\n\n"
                f"Анализ ошибок решения:\n{json.dumps(state.analysis_of(parent), ensure_ascii=False)[:2500]}\n\n"
                f"{state.tree.summary(parent)}\n\nДоступные действия:\n{menu}")
        answer = llm.ask_json(PLANNER_SYSTEM, user)
        if answer and answer.get("action") in options:
            log.info(f"LLM выбирает '{answer['action']}' для #{parent.id}: {answer.get('why', '')}")
            return answer["action"]
    for name in ("target_log", "class_balance"):
        if name in options:
            return name
    if "merge" in options and len(state.tree.nodes) >= 4:
        return "merge"
    if "tune_lgbm" in options and not _tuned_anywhere(state) and len(state.tree.nodes) >= 8:
        return "tune_lgbm"
    if "features" in options:
        return "features"
    for name in ("merge", "drop_weak", "tune_lgbm"):
        if name in options and (name != "tune_lgbm" or not _tuned_anywhere(state)):
            return name
    return None
