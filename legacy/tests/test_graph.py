"""
Тесты для LangGraph графа (agent.graph):
Проверяет разделение ответственности:
- Этап 1: profiling_node (жесткий код);
- Этап 2: meta_analysis_node (АГЕНТ: LLM Qwen);
- Этап 3: splitting_node (жесткий код);
- Сквозной запуск графа app.invoke() и через orchestrator.run_pipeline().
"""

from __future__ import annotations

from pathlib import Path
import pytest
pd = pytest.importorskip("pandas")
pytest.importorskip("langgraph")

from agent.graph import app as pipeline_graph, make_initial_state, profiling_node, splitting_node
from agent.orchestrator import AutoMLOrchestrator

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_profiling_node_is_rigid_code() -> None:
    """Узел 1: Чистый детерминированный код без LLM."""
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": ["1 000 ₽", "2 000 ₽", "3 000 ₽"],
        "target": [0, 1, 0],
    })
    state = make_initial_state(df=df, target="target")
    result = profiling_node(state)

    assert "clean_df" in result
    assert "column_types" in result
    assert "report" in result
    assert result["column_types"]["salary"] == "numeric"
    assert result["report"].n_rows == 3


def test_graph_pipeline_execution_end_to_end() -> None:
    """Сквозной прогон графа: profiling -> meta_analysis (агент) -> splitting."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": ["A", "B", "A", "B", "A", "B"],
        "label": [0, 1, 0, 1, 0, 1],
    })

    initial_state = make_initial_state(df=df, target="label", test_size=0.33, val_size=0.0)
    final_state = pipeline_graph.invoke(initial_state)

    # 1. Проверяем Этап 1
    assert final_state["clean_df"] is not None
    assert final_state["report"] is not None

    # 2. Проверяем Этап 2 (Агент)
    decision = final_state["decision"]
    assert decision is not None
    assert decision.task_type == "binary_classification"
    assert decision.draft_model.algorithm is not None
    assert decision.battle_model.algorithm is not None

    # 3. Проверяем Этап 3
    splits = final_state["splits"]
    assert splits is not None
    assert len(splits.train) > 0
    assert len(splits.test) > 0


def test_orchestrator_run_pipeline() -> None:
    """Проверка запуска сквозного графа через метод orchestrator.run_pipeline()."""
    df = pd.DataFrame({
        "x": [10, 20, 30, 40, 50, 60, 70, 80],
        "y": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    })

    orchestrator = AutoMLOrchestrator(target="y")
    final_state = orchestrator.run_pipeline(df, test_size=0.25, val_size=0.0)

    assert orchestrator.clean_df is not None
    assert orchestrator.report is not None
    assert orchestrator.decision is not None
    assert orchestrator.decision.task_type == "regression"
    assert orchestrator.splits is not None
    assert len(orchestrator.splits.train) == 6
    assert len(orchestrator.splits.test) == 2


@pytest.mark.skipif(not (DATA_RAW_DIR / "titanic.csv").exists(), reason="titanic.csv не найден")
def test_graph_on_real_titanic_dataset() -> None:
    """Прогон полного графа на реальном Titanic."""
    df = pd.read_csv(DATA_RAW_DIR / "titanic.csv")
    target = "survived" if "survived" in df.columns else "Survived"

    orchestrator = AutoMLOrchestrator(target=target)
    final_state = orchestrator.run_pipeline(df)

    assert orchestrator.decision.task_type == "binary_classification"
    assert orchestrator.splits.strategy == "stratified"
    assert len(orchestrator.splits.train) > 800
