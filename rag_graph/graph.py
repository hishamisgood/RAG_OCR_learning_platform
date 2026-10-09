"""Wires the nodes into the LangGraph state machine from the diagram."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .nodes import Deps, make_nodes
from .state import GraphState

# Each Self-RAG loop-back is ~6 node visits; leave headroom above LangGraph's default of 25.
RUN_CONFIG = {"recursion_limit": 60}


def build_graph(deps: Deps, s: Settings):
    nodes = make_nodes(deps, s)
    g = StateGraph(GraphState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "user_query")
    g.add_edge("user_query", "supervisor")
    g.add_edge("supervisor", "retriever")
    g.add_edge("retriever", "crag_evaluator")

    # CRAG gate: relevant docs go straight to generation, otherwise fall back to the web.
    g.add_conditional_edges(
        "crag_evaluator",
        lambda st: st["relevance"],
        {"relevant": "generator", "irrelevant": "web_search"},
    )
    g.add_edge("web_search", "generator")
    g.add_edge("generator", "selfrag_evaluator")

    # Self-RAG gate: pass -> output; hallucinated -> loop back to the Supervisor (bounded).
    def after_selfrag(st: GraphState) -> str:
        if st["grounded"] == "passed" or st.get("retries", 0) >= s.max_retries:
            return "output"
        return "retry"

    g.add_conditional_edges("selfrag_evaluator", after_selfrag, {"output": "output", "retry": "supervisor"})
    g.add_edge("output", END)
    return g.compile()


def run_question(graph, question: str) -> dict:
    """Run the graph to completion and return the final state."""
    return graph.invoke({"question": question}, config=RUN_CONFIG)
