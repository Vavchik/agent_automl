"""Р4. Дерево решений для AutoResearch (идея AIDE, Weco AI, arXiv:2502.13138).

Каждый узел — полное решение: признаки (Dataset) + настройки обучения (ModelConfig) + честный скор.
Корень — бейзлайн. Новая гипотеза — ребёнок какого-то узла, а не следующего звена цепочки.

Отличия от линейного цикла:
- откат: если ветка перестала улучшаться, поиск расширяет другие хорошие узлы, а не «портит» лучший;
- исследование: с вероятностью EXPLORE_PROB расширяем не лучший узел, а один из топ-3;
- слияние: признаки из двух удачных веток можно объединить (действие merge);
- у каждого узла не больше MAX_CHILDREN детей — неудачные узлы перестают забирать время.

Что улучшать (Р4): выбор узла по UCB вместо epsilon-greedy, учёт стоимости по времени,
черновики (несколько независимых идей от корня в начале).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import config
from core.contracts import Dataset, ExperimentResult, ModelConfig
from task.metrics import improvement


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    action: str
    hypothesis: str
    result: Optional[ExperimentResult]
    ds: Optional[Dataset]                    # None — данные не храним, узел нельзя расширять
    cfg: Optional[ModelConfig]
    depth: int = 0
    lineage: frozenset = frozenset()         # действия на пути от корня (для одноразовых действий)
    added_cols: list[str] = field(default_factory=list)   # колонки, которые узел добавил к родителю
    gain: float = 0.0                        # прирост к родителю (> 0 — лучше родителя)
    improving: bool = False                  # прирост к родителю устойчив (правило Р5)
    children: int = 0
    exhausted: bool = False                  # для узла больше нечего пробовать
    skipped: set = field(default_factory=set)  # действия, которым от этого узла нечего предложить
    code: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.error


class SolutionTree:
    def __init__(self, root: Node, metric: str, seed: int = config.SEED):
        self.metric = metric
        self.nodes: dict[int, Node] = {root.id: root}
        self.best_id = root.id
        self.rng = random.Random(seed)

    def new_id(self) -> int:
        return max(self.nodes) + 1

    def add(self, node: Node) -> None:
        self.nodes[node.id] = node
        if node.parent_id is not None:
            self.nodes[node.parent_id].children += 1

    @property
    def best(self) -> Node:
        return self.nodes[self.best_id]

    def ranked(self) -> list[Node]:
        """Успешные узлы от лучшего к худшему."""
        ok = [n for n in self.nodes.values() if n.ok]
        return sorted(ok, key=lambda n: -improvement(self.metric, n.result.score, self.best.result.score))

    def expandable(self) -> list[Node]:
        """Узлы, от которых можно строить дальше: корень и узлы с устойчивым приростом к родителю.
        Узлы без прироста — тупики: их не расширяем (это и есть откат к лучшим веткам)."""
        return [n for n in self.ranked()
                if (n.parent_id is None or n.improving)
                and n.ds is not None and not n.exhausted and n.children < config.MAX_CHILDREN]

    def select_parent(self) -> Optional[Node]:
        candidates = self.expandable()
        if not candidates:
            return None
        if len(candidates) > 1 and self.rng.random() < config.EXPLORE_PROB:
            return self.rng.choice(candidates[:3])
        return candidates[0]

    def merge_sources(self, target: Node) -> list[Node]:
        """Удачные узлы из других веток, чьих признаков ещё нет у target."""
        out = []
        own_path = {p.id for p in self.path(target)}
        for n in self.ranked():
            if n.id in own_path or n.parent_id == target.id:   # свои предки и дети — не «другие ветки»
                continue
            if not n.improving or not n.added_cols or n.ds is None:
                continue
            if any(c not in target.ds.X.columns for c in n.added_cols):
                out.append(n)
        return out

    def path(self, node: Node) -> list[Node]:
        chain = [node]
        while chain[-1].parent_id is not None:
            chain.append(self.nodes[chain[-1].parent_id])
        return chain[::-1]

    def prune(self, keep: int = config.KEEP_NODES) -> None:
        """Храним данные только у лучших узлов, чтобы не съесть память."""
        ranked = self.ranked()
        keep_ids = {n.id for n in ranked[:keep]} | {self.best_id}
        for n in self.nodes.values():
            if n.id not in keep_ids and n.parent_id is not None:
                n.ds = None

    def summary(self, focus: Node, last_failures: int = 8) -> str:
        """Текст для LLM: путь к узлу, который расширяем, и что не сработало."""
        lines = ["Путь к текущему решению (от бейзлайна):"]
        for n in self.path(focus):
            lines.append(f"  #{n.id} [{n.action}] {n.hypothesis[:160]} -> {n.result.score:.5f}")
        lines.append(f"Лучший скор во всём дереве: {self.best.result.score:.5f} (узел #{self.best_id})")
        tried = [n for n in self.nodes.values() if n.parent_id is not None and (not n.improving)]
        if tried:
            lines.append("Не сработало (не повторяй):")
            for n in tried[-last_failures:]:
                outcome = f"ошибка: {n.error[:120]}" if n.error else f"прирост {n.gain:+.5f}"
                lines.append(f"  #{n.id} [{n.action}] {n.hypothesis[:140]} -> {outcome}")
        good = [n for n in self.nodes.values() if n.improving and n.id not in {p.id for p in self.path(focus)}]
        if good:
            lines.append("Сработало в других ветках:")
            for n in good[-5:]:
                lines.append(f"  #{n.id} [{n.action}] {n.hypothesis[:140]} -> прирост {n.gain:+.5f}")
        return "\n".join(lines)
