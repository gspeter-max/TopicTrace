"""
Conditional edge functions for the RAG LangGraph.

Each function receives the current state and returns the NAME of the next
node to route to. LangGraph uses these to determine the path at runtime.
"""

from typing import Literal

from topictrace.rag.documentRetrieve.graph.state import RAGState


class StateWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        if isinstance(self._obj, dict):
            return self._obj.get(name)
        return getattr(self._obj, name)


def route_after_vector_search(
    state: RAGState,
) -> Literal["grade_chunks", "graph_search"]:
    """
    After vector search:
    - complex queries go straight to graph_search
    - simple queries go to grade_chunks first
    """
    state = StateWrapper(state)
    if state.intent == "complex":
        return "graph_search"
    return "grade_chunks"


def route_after_grader(state: RAGState) -> Literal["answer_node", "graph_search"]:
    """
    After grade_chunks (simple path only):
    - sufficient=True  → skip graph, go directly to answer_node (fast path)
    - sufficient=False → escalate to graph_search
    """
    state = StateWrapper(state)
    if state.grade_sufficient:
        return "answer_node"
    return "graph_search"
