from __future__ import annotations 

from typing import Any
from langgraph.graph import StateGraph, END

from schemas.state import AppState
from agents.planner import run_planner
from agents.researcher import run_research
from agents.writer import run_writer
from agents.verifier import run_verifier, should_reroute_to_research


def build_graph():
    graph = StateGraph(AppState)

    # Register workflow nodes
    graph.add_node("planner", run_planner)
    graph.add_node("research", run_research)
    graph.add_node("writer", run_writer)
    graph.add_node("verifier", run_verifier)

    # Define execution flow
    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "verifier")

    # Conditional routing after verification
    graph.add_conditional_edges(
        "verifier",
        should_reroute_to_research,
        {"research": "research", "end": END},
    )

    return graph.compile()


def _ensure_app_state(x: Any) -> AppState:
    # Ensure the result is returned as AppState
    if isinstance(x, AppState):
        return x
    if isinstance(x, dict):
        return AppState(**x)
    raise TypeError(f"Unexpected state type: {type(x)}")


def run_task(user_task: str, persist_dir: str = "data/chroma", model: str = "gpt-4o-mini") -> AppState:
    app = build_graph()
    state = AppState(user_task=user_task, meta={"persist_dir": persist_dir, "model": model})
    result = app.invoke(state)
    return _ensure_app_state(result)
