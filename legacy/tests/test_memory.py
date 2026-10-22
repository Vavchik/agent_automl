"""
Тесты для модуля памяти экспериментов (agent/memory.py).
"""

from agent.memory import ExperimentMemory


def test_experiment_memory_tracking():
    mem = ExperimentMemory(metric="macro_f1", higher_is_better=True)
    assert mem.get_best_score() is None
    assert "пуста" in mem.format_history_for_llm()

    # Log an accepted experiment
    rec1 = mem.log_experiment(
        iteration=1,
        name="log_transform",
        description="Взятие log1p от признаков с длинным хвостом",
        code="def transform_features(df): return df",
        score=0.825,
        baseline_score=0.810,
        status="accepted",
    )
    assert rec1.delta == 0.015
    assert mem.get_best_score() == 0.825
    assert len(mem.get_accepted_records()) == 1

    # Log a rejected experiment
    rec2 = mem.log_experiment(
        iteration=2,
        name="bad_features",
        description="Генерация полиномиальных признаков степени 3",
        code="def transform_features(df): return df",
        score=0.795,
        baseline_score=0.825,
        status="rejected",
    )
    assert rec2.delta == -0.03
    assert mem.get_best_score() == 0.825

    # Log a failed experiment
    rec3 = mem.log_experiment(
        iteration=3,
        name="syntax_error",
        description="Деление на 0 в признаке",
        code="bad code",
        score=None,
        baseline_score=0.825,
        status="failed",
        error_message="ZeroDivisionError: division by zero",
    )
    assert rec3.error_message is not None

    summary = mem.format_history_for_llm()
    assert "ПРИНЯТО" in summary
    assert "ОТКЛОНЕНО" in summary
    assert "ОШИБКА В КОДЕ" in summary
    assert "ZeroDivisionError" in summary
