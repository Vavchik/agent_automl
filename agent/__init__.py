"""
Пакет agent: оркестрация, принятие решений (мета-анализ), LangGraph граф и LLM.
"""

from agent.llm import ask_llm, extract_code, extract_json
from agent.meta_analyst import select_model_and_metric

try:
    from agent.graph import app as pipeline_graph, make_initial_state, PipelineState
    from agent.orchestrator import AutoMLOrchestrator
except ImportError:
    pipeline_graph = None
    make_initial_state = None
    PipelineState = None
    AutoMLOrchestrator = None

__all__ = [
    "ask_llm",
    "extract_code",
    "extract_json",
    "select_model_and_metric",
    "pipeline_graph",
    "PipelineState",
    "make_initial_state",
    "AutoMLOrchestrator",
]
