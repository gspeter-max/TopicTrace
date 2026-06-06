"""
Conditional edge functions for the RAG LangGraph.

Each function receives the current state and returns the NAME of the next
node to route to. LangGraph uses these to determine the path at runtime.
"""
from typing import Literal
from documentRetrieve.graph.state import RAGState


def route_after_vector_search(state: RAGState) -> Literal["grade_chunks", "graph_search"]:
    """
    After vector search:
    - complex queries go straight to graph_search
    - simple queries go to grade_chunks first
    """
    if state.get("intent") == "complex":
        return "graph_search"
    return "grade_chunks"


def route_after_grader(state: RAGState) -> Literal["answer_node", "graph_search"]:
    """
    After grade_chunks (simple path only):
    - sufficient=True  → skip graph, go directly to answer_node (fast path)
    - sufficient=False → escalate to graph_search
    """
    if state.get("grade_sufficient", False):
        return "answer_node"
    return "graph_search"
