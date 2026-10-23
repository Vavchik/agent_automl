"""Р4. Solution Tree (дерево решений) для SOTA исследования в пространстве кода.

В отличие от линейной цепочки (1 -> 2 -> 3), агент строит дерево решений:
- Каждый узел хранит состояние (Dataset, ModelConfig, ExperimentResult, статус);
- При падении или деградации скора агент выполняет Backtracking к лучшему узлу;
- Топовые узлы дерева сохраняются для последующего блендинга/стекинга.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.contracts import Dataset, ExperimentResult, ModelConfig
from runtime.logger import log


@dataclass
class SolutionNode:
    id: int
    parent_id: Optional[int]
    iteration: int
    action: str
    hypothesis: str
    ds: Dataset
    cfg: ModelConfig
    result: Optional[ExperimentResult] = None
    status: str = "pending"  # "pending" | "accepted" | "rejected" | "failed"
    code: str = ""
    error: str = ""
    depth: int = 0


class SolutionTree:
    def __init__(self, root_ds: Dataset, root_cfg: ModelConfig, root_result: ExperimentResult):
        self.nodes: dict[int, SolutionNode] = {}
        self.next_id = 0
        root_node = SolutionNode(
            id=self.next_id,
            parent_id=None,
            iteration=0,
            action="baseline",
            hypothesis="Baseline model on clean initial features",
            ds=root_ds,
            cfg=root_cfg,
            result=root_result,
            status="accepted",
            depth=0,
        )
        self.nodes[self.next_id] = root_node
        self.root_id = self.next_id
        self.best_id = self.next_id
        self.next_id += 1

    def add_candidate(self, parent_id: int, iteration: int, action: str, hypothesis: str,
                      ds: Dataset, cfg: ModelConfig, code: str = "") -> SolutionNode:
        parent = self.nodes.get(parent_id, self.nodes[self.root_id])
        node = SolutionNode(
            id=self.next_id,
            parent_id=parent.id,
            iteration=iteration,
            action=action,
            hypothesis=hypothesis,
            ds=ds,
            cfg=cfg,
            code=code,
            depth=parent.depth + 1,
        )
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node

    def record_result(self, node_id: int, result: ExperimentResult, accepted: bool, reason: str) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.result = result
        if accepted:
            node.status = "accepted"
            self.best_id = node_id
            log.info(f"[SolutionTree] Узел #{node_id} ({node.action}) ПРИНЯТ как новый лучший! (глубина {node.depth})")
        else:
            node.status = "rejected"
            log.info(f"[SolutionTree] Узел #{node_id} ({node.action}) отклонен. Backtracking к лучшему узлу #{self.best_id}")

    def record_failure(self, node_id: int, error: str) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.status = "failed"
        node.error = error
        log.warning(f"[SolutionTree] Узел #{node_id} ({node.action}) упал: {error[:150]}. Backtracking к лучшему узлу #{self.best_id}")

    def get_best_node(self) -> SolutionNode:
        return self.nodes[self.best_id]

    def get_exploration_node(self) -> SolutionNode:
        """Возвращает узел, от которого следует строить следующую гипотезу.
        При деградации или ошибке автоматически откатывается к лучшему узлу."""
        return self.get_best_node()

    def get_top_nodes(self, k: int = 5) -> list[SolutionNode]:
        """Возвращает до k лучших принятых узлов с валидационными результатами."""
        accepted = [n for n in self.nodes.values() if n.status == "accepted" and n.result is not None]
        unique: list[SolutionNode] = []
        seen_scores = set()
        for n in accepted:
            if n.result is None:
                continue
            sc = round(float(n.result.score), 6)
            if sc not in seen_scores:
                seen_scores.add(sc)
                unique.append(n)
        return sorted(unique, key=lambda n: n.result.score if n.result else 0.0, reverse=True)[:k]
