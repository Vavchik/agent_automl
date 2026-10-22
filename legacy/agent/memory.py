"""
Модуль долговременной и контекстной памяти агента (ExperimentMemory).

Хранит историю гипотез, сгенерированного кода, CV-метрик и ошибок.
Используется LLM для рефлексии (Reflection): что сработало, что ухудшило скор,
какие ошибки возникли в песочнице.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Optional

logger = logging.getLogger("agent.memory")


@dataclass
class ExperimentRecord:
    """Запись об одном исследовательском эксперименте."""
    iteration: int
    name: str
    description: str
    code: str
    score: Optional[float]
    baseline_score: float
    delta: float = 0.0
    status: str = "pending"  # "accepted", "rejected", "failed"
    error_message: Optional[str] = None
    duration_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ExperimentMemory:
    """
    Память экспериментов агента.
    Сохраняет цепочку итераций Feature Engineering и предоставляет
    сводку для промпта рефлексии.
    """

    def __init__(self, metric: str = "macro_f1", higher_is_better: bool = True):
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.records: list[ExperimentRecord] = []
        self.best_score: Optional[float] = None
        self.best_code_snippets: list[str] = []

    def log_experiment(
        self,
        iteration: int,
        name: str,
        description: str,
        code: str,
        score: Optional[float],
        baseline_score: float,
        status: str,
        error_message: Optional[str] = None,
        duration_sec: float = 0.0,
    ) -> ExperimentRecord:
        """Регистрирует завершенный эксперимент."""
        delta = 0.0
        if score is not None:
            delta = round(score - baseline_score, 6)

        record = ExperimentRecord(
            iteration=iteration,
            name=name,
            description=description,
            code=code,
            score=score,
            baseline_score=baseline_score,
            delta=delta,
            status=status,
            error_message=error_message,
            duration_sec=duration_sec,
        )
        self.records.append(record)

        if status == "accepted" and score is not None:
            self.best_score = score
            self.best_code_snippets.append(code)

        logger.info(
            f"[Memory] Эксперимент #{iteration} '{name}': {status.upper()} "
            f"(Score: {score}, Delta: {delta:+.5f})"
        )
        return record

    def get_best_score(self) -> Optional[float]:
        return self.best_score

    def get_accepted_records(self) -> list[ExperimentRecord]:
        return [r for r in self.records if r.status == "accepted"]

    def format_history_for_llm(self, max_records: int = 5) -> str:
        """
        Форматирует последние эксперименты для передачи в контекст LLM.
        Показывает, какие гипотезы дали прирост, а какие привели к ошибке или падению скора.
        """
        if not self.records:
            return "История экспериментов пуста. Это первая итерация."

        recent = self.records[-max_records:]
        lines = ["### Предыдущие эксперименты и рефлексия:"]

        for r in recent:
            status_symbol = "✅ ПРИНЯТО" if r.status == "accepted" else ("❌ ОТКЛОНЕНО" if r.status == "rejected" else "⚠️ ОШИБКА В КОДЕ")
            lines.append(f"\n- **Эксперимент #{r.iteration}**: {r.description}")
            lines.append(f"  • Статус: {status_symbol}")
            if r.score is not None:
                lines.append(f"  • Скор: {r.score:.5f} (изменение относительно бейзлайна: {r.delta:+.5f})")
            if r.error_message:
                clean_err = r.error_message.strip().split("\n")[-1]
                lines.append(f"  • Ошибка в исполнении: `{clean_err}`")

        lines.append(
            "\n**Указание для новой гипотезы:** "
            "Не повторяй идеи, которые привели к падению метрики или ошибке. "
            "Если идея была принята, развивай это направление или предложи принципиально новый доменный признак."
        )
        return "\n".join(lines)
