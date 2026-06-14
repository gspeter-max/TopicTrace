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

from langgraph.graph import END, START, StateGraph  # type: ignore[import] 

from topictrace.rag.documentRetrieve.graph.edges import (
    route_after_grader,
    route_after_vector_search,
)
from topictrace.rag.documentRetrieve.graph.nodes import (
    answer_node,
    grade_chunks_node,
    graph_search,
    rerank,
    route_query,
    vector_search,
)
from topictrace.rag.documentRetrieve.graph.state import RAGState


async def ragGraph():
    """
    Build and compile the RAG state machine.

    Returns a compiled LangGraph that can be called with:
        graph.ainvoke(initial_state, config={"configurable": {"neo4j_client": client}})
    """
    graph = (
        StateGraph(RAGState)
        .add_node("route_query", route_query)
        .add_node("vector_search", vector_search)
        .add_node("grade_chunks", grade_chunks_node)
        .add_node("graph_search", graph_search)
        .add_node("rerank", rerank)
        .add_node("answer_node", answer_node)
        # ── Wire edges ────────────────────────────────────────────────────────────
        .add_edge(START, "route_query")
        .add_edge("route_query", "vector_search")
        # After vector search: complex → graph_search, simple → grade_chunks
        .add_conditional_edges(
            "vector_search",
            route_after_vector_search,
            {"grade_chunks": "grade_chunks", "graph_search": "graph_search"},
        )
        # After grader: sufficient → answer_node, not sufficient → graph_search
        .add_conditional_edges(
            "grade_chunks",
            route_after_grader,
            {"answer_node": "answer_node", "graph_search": "graph_search"},
        )
        # Fixed downstream edges
        .add_edge("graph_search", "rerank")
        .add_edge("rerank", "answer_node")
        .add_edge("answer_node", END)
    )

    return graph.compile()
