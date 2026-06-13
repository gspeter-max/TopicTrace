"""
Retrieval pipeline entry point.

handle_query is the only public function.
It creates the Neo4j client, invokes the LangGraph, and maps
the final state back to QueryResponse.

All pipeline logic lives in documentRetrieve/graph/.
"""

from fastapi import Request

from topictrace import settings
from topictrace.db.neo4j import Neo4jClient
from topictrace.rag.documentRetrieve.graph.state import RAGState
from topictrace.server.schemas.rag.retrieveModels import QueryRequest, QueryResponse


async def handle_query(userInput: QueryRequest, r: Request) -> QueryResponse:
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
        final_state = await r.app.state.ragGraph.ainvoke(
            RAGState(
                query=userInput.query,
                top_k=userInput.top_k,
                top_k_rerank=userInput.top_k_rerank,
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
