"""
Retrieval pipeline entry point.

handle_query is the only public function.
It creates the Neo4j client, invokes the LangGraph, and maps
the final state back to QueryResponse.

All pipeline logic lives in documentRetrieve/graph/.
"""

from topictrace import settings
from topictrace.db.neo4j import Neo4jClient
from topictrace.rag.documentRetrieve.graph.build_graph import build_rag_graph
from topictrace.rag.documentRetrieve.graph.state import RAGState
from topictrace.server.schemas.rag.retrieveModels import QueryRequest, QueryResponse

# Build the graph once at module load — it is stateless and reusable
_rag_graph = build_rag_graph()


async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Entry point for the Hybrid Adaptive RAG pipeline.

    Creates the Neo4j client, invokes the LangGraph state machine,
    then closes the client regardless of success or failure.
    """
    client = Neo4jClient(
        settings.DATABASE_CONFIG.NEO4J.NEO4J_URI,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_USER,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD,
    )
    try:
        final_state = await _rag_graph.ainvoke(
            RAGState(
                query=request.query,
                top_k=request.top_k,
                top_k_rerank=request.top_k_rerank,
                intent="",
                raw_chunks=list(),
                vector_texts=list(),
                answer="",
                used_graph_search=False,
                reason_for_graph_search="",
                grade_sufficient=False,
                grade_answer="",
                grade_reason="",
                graph_facts="",
                final_context=[],
            ),
            config={"configurable": {"neo4j_client": client}},
            version="v2",
        )
    finally:
        await client.close()

    return QueryResponse(
        answer=final_state.value.answer,
        intent=final_state.value.intent,
        used_graph_search=final_state.value.used_graph_search,
        reason_for_graph_search=final_state.value.reason_for_graph_search,
        context_used=final_state.value.final_context,
    )
