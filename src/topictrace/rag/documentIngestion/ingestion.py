import asyncio
from collections import defaultdict
from typing import Any, Literal

from topictrace import log, settings
from topictrace.db.neo4j import Neo4jClient
from topictrace.db.neo4j.cypherQuerys import (
    save_chunk,
    save_document_node,
    save_entity_nodes_and_relationships,
)
from topictrace.provider.embedding import embeddingModel
from topictrace.provider.llm import get_llm
from topictrace.rag.documentIngestion.contextual_retrieval import (
    build_contextualized_document,
)
from topictrace.rag.documentIngestion.entityResolution import (
    find_fuzzy_merge_candidates,
    resolve_ambiguous_entity_pairs,
    split_clear_cases_from_ambiguous_cases,
)
from topictrace.rag.documentIngestion.graphExtraction import (
    extract_graph_data_from_chunk,
)
from topictrace.rag.documentIngestion.graphPersistence import (
    build_neo4j_graph_write_payload,
    rewrite_graph_results_to_canonical_entities,
)
from topictrace.rag.documentIngestion.models.graphExtractionModels import (
    ChunkGraphExtractionResult,
)


async def build_contextualized_chunk_embeddings(
    chunks: list[dict[str, Any]],
) -> list[list[float]]:
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
    llm_provider: Literal["MISTRAL_AI", "DEEPSEEK_AI"] = settings.DEFAULT_LLM_PROVIDER,
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
    return await asyncio.gather(
        *[
            extract_graph_data_from_chunk(llm_client=client, chunk=chunk)
            for chunk in chunks
        ]
    )


async def resolve_entities_for_graph(
    raw_chunk_graph_results: list[ChunkGraphExtractionResult],
    existing_entity: set[str] | None = None,
    llm_provider: Literal["MISTRAL_AI", "DEEPSEEK_AI"] = settings.DEFAULT_LLM_PROVIDER,
) -> dict[str, dict[str, Any]]:
    """
    Collapses raw entity name variants into one canonical name per real-world entity.

    Two-stage funnel:
    - Stage 1 (fuzzy): ``rapidfuzz.token_set_ratio ≥ FUZZY_THRESHOLD`` → candidate pairs.
    - Stage 2 (split): score ≥ HIGH_THRESHOLD → auto-merge; rest → LLM disambiguation.
    Existing DB entities (``existing_entity``) are excluded from new canonical creation.

    It creates and returns two mappings concurrently:
    1. canonical_name_by_raw_name: A dict mapping canonical names to their list of aliases.
    2. canonical_by_alias: A flat lookup dictionary mapping every entity name (canonical
       or alias) directly to its canonical name.

    Args:
        raw_chunk_graph_results: Output of ``extract_chunk_graph_data_in_parallel``.
        existing_entity: Canonical names already in Neo4j for this document, or None.
        llm_provider: LLM provider to use.
    Returns:
        A dictionary containing:
        - "canonical_name_by_raw_name": dict[str, list[str]]
        - "canonical_by_alias": dict[str, str]
    """
    unique_entities: set[str] = (
        existing_entity.copy() if existing_entity is not None else set()
    )
    for res in raw_chunk_graph_results:
        for e in res.entities:
            unique_entities.add(" ".join(e.entity_name.strip().split()))
        for r in res.relationships:
            unique_entities.add(" ".join(r.source_entity_name.strip().split()))
            unique_entities.add(" ".join(r.target_entity_name.strip().split()))

    if not unique_entities:
        return {"canonical_name_by_raw_name": {}, "canonical_by_alias": {}}

    canonical_map: dict[str, list[str]] = defaultdict(list)
    canonical_by_alias: dict[str, str] = {}

    try:
        fuzzy_candidates, left_candidates = find_fuzzy_merge_candidates(unique_entities)
        same_pairs, diff_entities = split_clear_cases_from_ambiguous_cases(
            [
                (l, r, settings.ENTITY_RESOLUTION_DEFAULT_CANDIDATE_SCORE)
                for l, r in fuzzy_candidates
            ]
        )

        left_candidates = left_candidates.union(diff_entities)

        client = get_llm(llm_provider)
        decisions = await resolve_ambiguous_entity_pairs(
            llm_client=client, ambiguous_pairs=left_candidates
        )

        for decision in decisions:
            canonical_map[decision.canonical_name] = [
                decision.left_name,
                decision.right_name,
            ]
            canonical_by_alias[decision.canonical_name] = decision.canonical_name
            canonical_by_alias[decision.left_name] = decision.canonical_name
            canonical_by_alias[decision.right_name] = decision.canonical_name

        left_candidates = left_candidates.difference(set(canonical_map.keys()))
        if existing_entity is not None:
            left_candidates = left_candidates.difference(existing_entity)

        for entity in left_candidates:
            canonical_map[entity] = []
            canonical_by_alias[entity] = entity

        for l, r, _ in same_pairs:
            if l in canonical_map.keys():
                canonical_map[l].append(r)
                canonical_by_alias[r] = l
            elif r in canonical_map.keys():
                canonical_map[r].append(l)
                canonical_by_alias[l] = r
            else:
                canonical_map[l] = [r]
                canonical_by_alias[l] = l
                canonical_by_alias[r] = l

    except Exception as exc:
        log.error(
            "Error occurred during entity resolution. Falling back to identity mapping.",
            error=str(exc),
        )
        canonical_map.clear()
        canonical_by_alias.clear()
        for entity in unique_entities:
            canonical_map[entity] = []
            canonical_by_alias[entity] = entity

    return {
        "canonical_name_by_raw_name": dict(canonical_map),
        "canonical_by_alias": canonical_by_alias,
    }


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
            neo4j_client, doc["document_id"], doc.get("source_file", "")
        )

        chunks = doc["chunks"]
        chunk_to_ids_map = graph_write_payload.get("chunk_to_entity_ids", {})

        # Save all chunks at once, each with its own list of name codes
        await asyncio.gather(
            *[
                save_chunk(
                    neo4j_client,
                    chunk,
                    embedding,
                    entity_ids=chunk_to_ids_map.get(chunk["chunk_id"], []),
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
        )

        # Save the graph nodes and connections
        await save_entity_nodes_and_relationships(neo4j_client, graph_write_payload)

    finally:
        await neo4j_client.close()


def build_ingestion_summary_response(
    contextualized_document: dict[str, Any], canonical_graph_payload: Any
) -> dict[str, int | str]:
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
        "canonical_entity_count": len(
            set(e["canonical_name"] for e in canonical_graph_payload.entities)
        ),
        "relationship_count": len(canonical_graph_payload.relationships),
    }


async def get_neo4j_entities_by_document(
    client: Neo4jClient, document_id: str
) -> set[str]:
    query = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
    WHERE c.document_id = $document_id
    RETURN DISTINCT e.canonical_name AS entity_name
    """
    records = await client.execute_query(query, {"document_id": document_id})
    return {record["entity_name"] for record in records}


async def ingest_document_graph(
    file_path: str,
    provider: Literal["MISTRAL_AI", "DEEPSEEK_AI"] = settings.DEFAULT_LLM_PROVIDER,
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
    log.info(
        "Document parsed and chunked",
        num_chunks=len(doc["chunks"]),
        document_id=doc["document_id"],
    )

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
        client=neo4j_client, document_id=doc["document_id"]
    )
    resolution_result = await resolve_entities_for_graph(
        raw_results, existing_entity=neo4j_entitys
    )
    log.info(
        "Entity resolution complete",
        num_canonical_names=len(resolution_result["canonical_name_by_raw_name"]),
    )

    canonical_payload = rewrite_graph_results_to_canonical_entities(
        raw_results,
        resolution_result["canonical_name_by_raw_name"],
        canonical_by_alias=resolution_result.get("canonical_by_alias"),
    )

    graph_write_payload = build_neo4j_graph_write_payload(
        doc["document_id"],
        canonical_payload.entities,
        canonical_payload.relationships,
    )
    log.info("Neo4j graph write payload prepared")

    await persist_document_graph(doc, embeddings, graph_write_payload)
    log.info("Document successfully persisted to Neo4j graph database")

    return build_ingestion_summary_response(doc, canonical_payload)
