"""Р4. Журнал экспериментов: память для LLM и материал для защиты.

Каждая запись сразу дописывается в experiments.jsonl рядом с task.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Record:
    iteration: int
    action: str
    hypothesis: str
    score: float | None
    best_before: float | None
    accepted: bool
    reason: str
    seconds: float
    error: str = ""
    code: str = ""


class Memory:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: list[Record] = []

    def log(self, record: Record) -> None:
        self.records.append(record)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

    def tried(self, action: str) -> int:
        return sum(r.action == action for r in self.records)

    def summary(self, last: int = 10) -> str:
        lines = []
        for r in self.records[-last:]:
            if r.error:
                outcome = f"ошибка: {r.error[:200]}"
            elif r.score is None:
                outcome = "пропущено"
            else:
                outcome = f"{r.score:.5f} — {'ПРИНЯТО' if r.accepted else 'отклонено'} ({r.reason})"
            lines.append(f"#{r.iteration} [{r.action}] {r.hypothesis[:200]} -> {outcome}")
        return "\n".join(lines) or "Экспериментов ещё не было."
