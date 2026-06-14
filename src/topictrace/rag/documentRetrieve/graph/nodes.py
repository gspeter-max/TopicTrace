"""
LangGraph node functions for the Hybrid Adaptive RAG pipeline.

Each node:
- Accepts (state: RAGState, config: RunnableConfig) or just (state: RAGState)
- Returns a PARTIAL dict of only the keys it modifies
- Never mutates state in-place

The Neo4j client is passed via config["configurable"]["neo4j_client"] so it
is created ONCE in handle_query and closed there in a finally block.
Business logic (classify_intent, grade_chunks, etc.) is unchanged.
"""

from typing import Any

from topictrace import log, settings
from topictrace.db.neo4j import Neo4jClient
from topictrace.db.neo4j.cypherQuerys import retrieve_similar_chunks
from topictrace.prompts import get_system_prompt
from topictrace.provider.embedding import embeddingModel
from topictrace.provider.llm import get_llm
from topictrace.provider.rerank import rerank_documents
from topictrace.rag.documentRetrieve.grader import grade_chunks
from topictrace.rag.documentRetrieve.graph.state import RAGState
from topictrace.rag.documentRetrieve.graphAgent import gather_graph_facts
from topictrace.rag.documentRetrieve.router import classify_intent


def _get_neo4j_client() -> Neo4jClient:
    """Extract the shared Neo4j client from LangGraph config or create a fallback one."""
    return Neo4jClient(
        settings.DATABASE_CONFIG.NEO4J.NEO4J_URI,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_USER,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD,
    )


def _extract_entity_ids(chunks: list[dict[str, Any]]) -> list[str]:
    """Deduplicate entity IDs from a list of chunk dicts."""
    entity_ids: set[str] = set()
    for chunk in chunks:
        ids = chunk.get("entity_ids")
        if isinstance(ids, list):
            entity_ids.update(ids)  # update(iter(list[]))
    return list(entity_ids)


# ── Node 1: route_query ───────────────────────────────────────────────────────
async def route_query(state: RAGState | str) -> dict[str, str]:
    """Classify the query as 'simple' or 'complex' using the LLM router."""
    query = state.query if isinstance(state, RAGState) else state

    intent = await classify_intent(query)
    log.info("Query intent classified", intent=intent, query=query)
    return {"intent": intent}

# ── Node 2: vector_search ─────────────────────────────────────────────────────


async def vector_search(state: RAGState) -> dict[str, Any]:
    """Embed the query and run vector similarity search in Neo4j."""
    client = _get_neo4j_client()

    embedder = embeddingModel(
        api_key=settings.EMBEDDING_CONFIG.JINA_API_KEY,
        base_url=settings.EMBEDDING_CONFIG.JINA_BASE_URL,
        embeddingModel=settings.EMBEDDING_CONFIG.JINA_EMBEDDING_MODEL,
        max_concurrency=settings.EMBEDDING_CONFIG.MAX_CONCURRENCY,
    )
    query_embedding = await embedder.generateEmebedding(state.query)

    raw_chunks = await retrieve_similar_chunks(
        client=client,
        index_name=settings.NEO4J_INDEX_NAME,
        query_embedding=query_embedding,
        top_k=state.top_k,
    )

    vector_texts = [c["full_context"] for c in raw_chunks]
    log.info("Vector search complete", num_chunks=len(vector_texts))

    return {"raw_chunks": raw_chunks, "vector_texts": vector_texts}

# ── Node 3: grade_chunks ──────────────────────────────────────────────────────


async def grade_chunks_node(state: RAGState) -> dict[str, Any]:
    """Grade whether vector chunks are sufficient to answer the query."""
    result = await grade_chunks(state.query, state.vector_texts)
    log.info("Grader result", sufficient=result.sufficient, reason=result.reason)
    return {
        "grade_sufficient": result.sufficient,
        "grade_reason": result.reason,
        "grade_answer": result.answer,
    }


# ── Node 4: graph_search ──────────────────────────────────────────────────────


async def graph_search(state: RAGState) -> dict[str, Any]:
    """Traverse the Neo4j knowledge graph using entity IDs from vector chunks."""
    client = _get_neo4j_client()
    entity_ids = _extract_entity_ids(state.raw_chunks)
    graph_facts = await gather_graph_facts(client, entity_ids)

    # reason_for_graph_search is only meaningful on the escalation path
    log.info(
        "Graph search complete",
        entity_count=len(entity_ids),
        has_facts=bool(graph_facts),
    )
    return {
        "graph_facts": graph_facts,
        "used_graph_search": True,
        "reason_for_graph_search": state.grade_reason,
    }


# ── Node 5: rerank ────────────────────────────────────────────────────────────


async def rerank(state: RAGState) -> dict[str, Any]:
    """Rerank combined vector + graph context using Voyage AI."""

    context_to_rerank: list[Any] = list(state.vector_texts)
    graph_facts = state.graph_facts
    if graph_facts:
        context_to_rerank.append(graph_facts)

    log.info("Reranking context", num_items=len(context_to_rerank))
    final_context = await rerank_documents(
        query=state.query,
        documents=context_to_rerank,
        top_k=state.top_k_rerank,
    )
    return {"final_context": final_context}


# ── Node 6: answer_node ───────────────────────────────────────────────────────


async def answer_node(state: RAGState) -> dict[str, Any]:
    """
    Generate the final answer.

    Fast path: if grader already generated an answer (grade_sufficient=True),
    return it immediately without calling the LLM again.

    Standard path: call Mistral with the reranked context.
    """
    # Fast path — grader pre-generated the answer
    if state.grade_sufficient:
        log.info("Fast path: using grader pre-generated answer")
        return {
            "answer": state.grade_answer,
            "final_context": state.vector_texts,
        }

    # Standard path — generate from reranked context
    context_texts: list[str] = state.final_context
    if not context_texts:
        return {"answer": "I could not find relevant information to answer your query."}

    context_block = "\n\n---\n\n".join(context_texts)
    prompt = get_system_prompt("answer_generator", {"context_block": context_block})

    try:
        llm = get_llm("MISTRAL_AI")
        bound_llm = llm.bind(temperature=0.0)
        response = await bound_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": state.query},
            ]
        )
        answer = response.content or "No answer generated."

    except Exception as e:
        log.error("Failed to generate final answer", error=str(e))
        answer = "Error generating answer from LLM."

    finally:
        log.info("Final answer generated")

    return {"answer": answer}
