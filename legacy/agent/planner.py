"""Р4. Выбор следующего действия.

Если LLM доступна и уже есть история, LLM выбирает действие из доступных и объясняет почему.
Иначе — фиксированный порядок DEFAULT_ORDER (дешёвые и частые выигрыши в начале).

Что улучшать (Р4): учитывать стоимость действия по времени, бюджет токенов,
останавливаться, когда несколько итераций подряд нет прироста.
"""
from __future__ import annotations

from agent.actions import ACTIONS, ResearchState
from agent.llm import llm
from runtime.logger import log

DEFAULT_ORDER = ["target_log", "class_balance", "features", "features",
                 "tune_lgbm", "features", "drop_weak", "features"]

PLANNER_SYSTEM = """Ты руководишь исследованием ML-модели для табличной задачи.
По истории экспериментов выбери ОДНО следующее действие из списка доступных.
Верни только JSON: {"action": "<имя>", "why": "<одно предложение>"}"""


def available_actions(state: ResearchState) -> list[str]:
    return [a.name for a in ACTIONS.values()
            if (a.repeatable or not state.memory.tried(a.name)) and a.can_run(state)]


def next_action(state: ResearchState) -> str | None:
    options = available_actions(state)
    if not options:
        return None
    if llm.available() and len(state.memory.records) >= 2:
        menu = "\n".join(f"- {name}: {ACTIONS[name].description}" for name in options)
        user = (f"Метрика {state.spec.metric}, лучший скор {state.best.score:.5f}, "
                f"осталось {state.clock.remaining() / 60:.0f} мин.\n\nДоступные действия:\n{menu}\n\n"
                f"История:\n{state.memory.summary()}")
        answer = llm.ask_json(PLANNER_SYSTEM, user, max_tokens=300)
        if answer and answer.get("action") in options:
            log.info(f"LLM выбирает '{answer['action']}': {answer.get('why', '')}")
            return answer["action"]
    for i, name in enumerate(DEFAULT_ORDER):
        if name in options and state.memory.tried(name) < DEFAULT_ORDER[: i + 1].count(name):
            return name
    return "features" if "features" in options else options[0]
