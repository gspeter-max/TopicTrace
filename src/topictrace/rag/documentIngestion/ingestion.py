import asyncio
from typing import Any, Literal
from collections import defaultdict
from topictrace.rag.documentIngestion.contextual_retrieval import build_contextualized_document
from topictrace.rag.documentIngestion.models.graphExtractionModels import ChunkGraphExtractionResult
from topictrace.provider.llm import get_llm
from topictrace.provider.embedding import embeddingModel
from topictrace.db.neo4j import Neo4jClient
from topictrace.db.neo4j.cypherQuerys import (
    save_chunk,
    save_document_node,
    save_entity_nodes_and_relationships
)
from topictrace.rag.documentIngestion.graphExtraction import extract_graph_data_from_chunk
from topictrace.rag.documentIngestion.entityResolution import (
    find_fuzzy_merge_candidates,
    split_clear_cases_from_ambiguous_cases,
    resolve_ambiguous_entity_pairs,
)
from topictrace.rag.documentIngestion.graphPersistence import (
    rewrite_graph_results_to_canonical_entities,
    build_neo4j_graph_write_payload,
)

from topictrace import log, settings

async def build_contextualized_chunk_embeddings(chunks: list[dict[str, Any]]) -> list[list[float]]:
    """
    Embeds each chunk's ``contextualized_text`` via Jina concurrently.

    Args:
        chunks: Chunk dicts from ``build_contextualized_document``; each must have ``contextualized_text``.
    Returns:
        Float vectors, same order as input. Empty list if chunks is empty.
    """
    embedding_model = embeddingModel(
        api_key=settings.EMBEDDING_CONFIG.JINA_API_KEY,
        base_url=settings.EMBEDDING_CONFIG.JINA_BASE_URL,
        embeddingModel=settings.EMBEDDING_CONFIG.JINA_EMBEDDING_MODEL,
        max_concurrency=settings.EMBEDDING_CONFIG.MAX_CONCURRENCY,
    )
    
    if not chunks:
        return []
    texts = [chunk["contextualized_text"] for chunk in chunks]
    return await embedding_model.generateEmebedding(texts)


async def extract_chunk_graph_data_in_parallel(
        chunks: list[dict[str, Any]], 
        llm_provider: Literal["MISTRAL_AI","DEEPSEEK_AI"]  = settings.DEFAULT_LLM_PROVIDER
    ) -> list[ChunkGraphExtractionResult]:
    """
    Runs LLM graph extraction over all chunks concurrently (shared client).

    Args:
        chunks: Contextualized chunk dicts.
        llm_provider: LLM provider to use.
    Returns:
        One ``ChunkGraphExtractionResult`` per chunk with raw ``.entities`` and ``.relationships``.
    """
    client = get_llm(llm_provider)
    return await asyncio.gather(*[
        extract_graph_data_from_chunk(llm_client=client, chunk=chunk)
        for chunk in chunks
    ])


async def resolve_entities_for_graph(
        raw_chunk_graph_results: list[Any], 
        existing_entity : set[str],
        llm_provider: Literal["MISTRAL_AI","DEEPSEEK_AI"]  = settings.DEFAULT_LLM_PROVIDER
    ) -> dict[str, dict[str, Any]]:

    """
    Collapses raw entity name variants into one canonical name per real-world entity.

    Two-stage funnel:
    - Stage 1 (fuzzy): ``rapidfuzz.token_set_ratio ≥ FUZZY_THRESHOLD`` → candidate pairs.
    - Stage 2 (split): score ≥ HIGH_THRESHOLD → auto-merge; rest → LLM disambiguation.
    Existing DB entities (``existing_entity``) are excluded from new canonical creation.

    Args:
        raw_chunk_graph_results: Output of ``extract_chunk_graph_data_in_parallel``.
        existing_entity: Canonical names already in Neo4j for this document.
        llm_provider: LLM provider to use.
    Returns:
        ``{"canonical_name_by_raw_name": {canonical: [raw, ...]}}`` covering all entity
        and relationship participants across all chunks.
    """

    unique_entitys: set[str]= existing_entity.copy() 
    for res in raw_chunk_graph_results:
        for e in res.entities:
            unique_entitys.add(" ".join(e.entity_name.strip().split()))
        for r in res.relationships:
            unique_entitys.add(" ".join(r.source_entity_name.strip().split()))
            unique_entitys.add(" ".join(r.target_entity_name.strip().split()))
            
    if not unique_entitys:
        return {"canonical_name_by_raw_name": {}}
        
    canonical_map : dict[str, list[str]] = defaultdict(list)
    fuzzy_candidates, left_candidates = find_fuzzy_merge_candidates(unique_entitys)
    same_pairs, diff_entitys = split_clear_cases_from_ambiguous_cases(
        [(l, r, settings.ENTITY_RESOLUTION_DEFAULT_CANDIDATE_SCORE) for l, r in fuzzy_candidates]
    )
    
    left_candidates: set[str] = left_candidates.union(diff_entitys) 

    client = get_llm(llm_provider)
    decisions = await resolve_ambiguous_entity_pairs(
        llm_client=client, 
        ambiguous_pairs= left_candidates
    ) 
    
    for decision in decisions:
        canonical_map[decision.canonical_name] = [ decision.left_name, decision.right_name ]

    left_candidates = left_candidates.difference(set(canonical_map.keys()))
    left_candidates = left_candidates.difference(existing_entity)
    
    for entity in left_candidates:
        canonical_map[entity] = [] 

    for l, r, _ in same_pairs:
        if l in canonical_map.keys(): 
            canonical_map[l].append(r)
        elif r in canonical_map.keys(): 
            canonical_map[r].append(l)
        else : 
            canonical_map[l] = [r]

    return {"canonical_name_by_raw_name": canonical_map}


async def persist_document_graph(
    doc: dict[str, Any],
    embeddings: list[list[float]],
    graph_write_payload: dict[str, Any],
) -> None:
    """
    Writes the processed document to Neo4j in 3 steps (client closed in ``finally``):

    1. MERGE ``:Document`` node.
    2. MERGE ``:Chunk`` nodes concurrently — with embedding vector + ``entity_ids``.
    3. MERGE ``:Entity`` nodes, ``:MENTIONED_IN`` and ``:RELATES_TO`` edges.

    Args:
        doc: Contextualized document dict; requires keys ``document_id``, ``source_file``, ``chunks``.
        embeddings: Float vectors, one per chunk, same order as ``doc["chunks"]``.
        graph_write_payload: Output of ``build_neo4j_graph_write_payload`` — entities, mentions,
                             relationships, chunk_to_entity_ids.
    """
    neo4j_client = Neo4jClient(
        settings.DATABASE_CONFIG.NEO4J.NEO4J_URI,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_USER,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD,
    )
    try: 
               
        await save_document_node(
            neo4j_client, 
            doc["document_id"], 
            doc.get("source_file", "")
        )

        chunks = doc["chunks"]
        chunk_to_ids_map = graph_write_payload.get("chunk_to_entity_ids", {})

        # Save all chunks at once, each with its own list of name codes
        await asyncio.gather(*[
            save_chunk(
                neo4j_client, 
                chunk, 
                embedding,
                entity_ids=chunk_to_ids_map.get(chunk["chunk_id"], [])
            )
            for chunk, embedding in zip(chunks, embeddings)
        ])
        
        # Save the graph nodes and connections
        await save_entity_nodes_and_relationships(neo4j_client, graph_write_payload)

    finally:
        await neo4j_client.close()


def build_ingestion_summary_response(contextualized_document: dict[str, Any], canonical_graph_payload: Any) -> dict[str, int | str]:
    """
    Builds the API response dict after ingestion.

    Args:
        contextualized_document: Unused; kept for consistent helper signature.
        canonical_graph_payload: ``CanonicalGraphPersistencePayload`` with final ``.entities`` / ``.relationships``.
    Returns:
        ``{status, message, raw_entity_count, canonical_entity_count, relationship_count}``.
    """
    return {
        "status": "success",
        "message": "Document ingested successfully",
        "raw_entity_count": len(canonical_graph_payload.entities),
        "canonical_entity_count": len(set(e["canonical_name"] for e in canonical_graph_payload.entities)),
        "relationship_count": len(canonical_graph_payload.relationships),
    }

async def get_neo4j_entities_by_document(client: Neo4jClient, document_id: str) -> set[str]:
    query = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
    WHERE c.document_id = $document_id
    RETURN DISTINCT e.canonical_name AS entity_name
    """
    records = await client.execute_query(query, {"document_id": document_id})
    return {record["entity_name"] for record in records}


async def ingest_document_graph(
    file_path: str, 
    provider: Literal["MISTRAL_AI","DEEPSEEK_AI"] = settings.DEFAULT_LLM_PROVIDER
) -> dict[str, int | str]:
    """
    Master orchestrator — runs the full document → Neo4j knowledge graph pipeline.

    Stages:
        1. parse + contextualize  → ``build_contextualized_document``
        2. embed chunks           → ``build_contextualized_chunk_embeddings``
        3. extract graph (LLM)    → ``extract_chunk_graph_data_in_parallel``
        4. resolve entity names   → ``resolve_entities_for_graph``
        5. build write payload    → ``build_neo4j_graph_write_payload``
        6. persist to Neo4j       → ``persist_document_graph``

    Args:
        file_path: Absolute path to the document to ingest.
        provider: ``'MISTRAL_AI'`` (default) or ``'DEEPSEEK_AI'``.
    Returns:
        Ingestion stats: ``{status, message, raw_entity_count, canonical_entity_count, relationship_count}``.
    """
    log.info("Starting document ingestion pipeline", file_path=file_path)
    
    llm = get_llm(provider)
    doc = await build_contextualized_document(file_path=file_path, client=llm)
    log.info("Document parsed and chunked", num_chunks=len(doc["chunks"]), document_id=doc["document_id"])
    
    embeddings = await build_contextualized_chunk_embeddings(doc["chunks"])
    log.info("Embeddings generated for chunks", num_embeddings=len(embeddings))
    
    raw_results = await extract_chunk_graph_data_in_parallel(doc["chunks"])
    log.info("Graph data extracted from chunks via LLM", num_results=len(raw_results))

    neo4j_client = Neo4jClient(
        settings.DATABASE_CONFIG.NEO4J.NEO4J_URI,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_USER,
        settings.DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD,
    )
    neo4j_entitys = await get_neo4j_entities_by_document(
        client=neo4j_client,
        document_id=doc["document_id"]
    )
    resolution_result = await resolve_entities_for_graph(
        raw_results, 
        existing_entity= neo4j_entitys
    )
    log.info("Entity resolution complete", num_canonical_names=len(resolution_result["canonical_name_by_raw_name"]))
    
    canonical_payload = rewrite_graph_results_to_canonical_entities(
        raw_results,
        resolution_result["canonical_name_by_raw_name"],
    )
    
    graph_write_payload = build_neo4j_graph_write_payload(
        doc["document_id"],
        canonical_payload.entities,
        canonical_payload.relationships,
    )
    log.info("Neo4j graph write payload prepared")
    
    await persist_document_graph(
        doc, 
        embeddings, 
        graph_write_payload
    )
    log.info("Document successfully persisted to Neo4j graph database")
    
    return build_ingestion_summary_response(doc, canonical_payload)
