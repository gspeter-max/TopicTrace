"""
Compile and return the Hybrid Adaptive RAG LangGraph.

Graph topology:
    START → route_query → vector_search
                              ↓ (complex)          ↓ (simple)
                         graph_search          grade_chunks
                              ↓                    ↓ (sufficient)    ↓ (not sufficient)
                           rerank             answer_node         graph_search
                              ↓                                        ↓
                         answer_node                               rerank
                              ↓                                        ↓
                             END                                  answer_node
                                                                       ↓
                                                                      END
"""
from langgraph.graph import StateGraph, START, END

from documentRetrieve.graph.state import RAGState
from documentRetrieve.graph.nodes import (
    route_query,
    vector_search,
    grade_chunks_node,
    graph_search,
    rerank,
    answer_node,
)
from documentRetrieve.graph.edges import route_after_vector_search, route_after_grader


def build_rag_graph():
    """
    Build and compile the RAG state machine.

    Returns a compiled LangGraph that can be called with:
        graph.ainvoke(initial_state, config={"configurable": {"neo4j_client": client}})
    """
    builder = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("route_query", route_query)
    builder.add_node("vector_search", vector_search)
    builder.add_node("grade_chunks", grade_chunks_node)
    builder.add_node("graph_search", graph_search)
    builder.add_node("rerank", rerank)
    builder.add_node("answer_node", answer_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    builder.add_edge(START, "route_query")
    builder.add_edge("route_query", "vector_search")

    # After vector search: complex → graph_search, simple → grade_chunks
    builder.add_conditional_edges(
        "vector_search",
        route_after_vector_search,
        {"grade_chunks": "grade_chunks", "graph_search": "graph_search"},
    )

    # After grader: sufficient → answer_node, not sufficient → graph_search
    builder.add_conditional_edges(
        "grade_chunks",
        route_after_grader,
        {"answer_node": "answer_node", "graph_search": "graph_search"},
    )

    # Fixed downstream edges
    builder.add_edge("graph_search", "rerank")
    builder.add_edge("rerank", "answer_node")
    builder.add_edge("answer_node", END)

    return builder.compile()
