import asyncio
from typing import Any

from topictrace.rag.documentIngestion.contextual_retrieval import build_contextualized_document
from topictrace.provider.llm import get_llm
from topictrace.provider.embedding import embeddingModel
from topictrace.db.neo4j import Neo4jClient
from topictrace.db.neo4j.cypherQuerys import (
    create_vector_index,
    save_chunk,
    save_document_node,
    save_entity_nodes_and_relationships
)
from topictrace.rag.documentIngestion.graphExtraction import extract_graph_data_from_chunk
from topictrace.rag.documentIngestion.entityResolution import (
    group_entities_by_normalized_name,
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
    Converts every chunk's contextualized text into a float vector (embedding)
    using the Jina embedding model. All chunks are embedded concurrently.

    Each chunk must have a "contextualized_text" key — this is the LLM-generated
    context prefix prepended to the raw chunk text (see contextual_retrieval.py).

    Args:
        chunks: List of chunk dicts from build_contextualized_document.

    Returns:
        A list of float vectors in the same order as the input chunks.
        Returns an empty list if no chunks are provided.
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


async def extract_chunk_graph_data_in_parallel(chunks: list[dict[str, Any]], provider: str = "MISTRAL_AI") -> list[Any]:
    """
    Runs the LLM-based graph extraction over every chunk in parallel.

    For each chunk, calls extract_graph_data_from_chunk which asks the LLM to
    identify entities (people, places, concepts) and relationships between them.
    All chunks share one LLM client instance to avoid repeated auth overhead.

    Args:
        chunks: List of contextualized chunk dicts.
        provider: Name of the LLM provider to use.

    Returns:
        List of ChunkGraphExtractionResult objects — one per chunk — each
        containing raw .entities and .relationships found by the LLM.
    """
    client = get_llm(provider)
    return await asyncio.gather(*[
        extract_graph_data_from_chunk(llm_client=client, chunk=chunk)
        for chunk in chunks
    ])


async def resolve_entities_for_graph(raw_chunk_graph_results: list[Any], provider: str = "MISTRAL_AI") -> dict[str, Any]:
    """
    Resolves raw entity names into a single canonical name per real-world entity.

    The LLM often extracts the same entity with slightly different names across
    chunks (e.g. "Apple Inc", "Apple", "apple inc"). This two-stage funnel
    collapses those variants into one agreed-upon name:

    Stage 1 — Exact normalization:
        Groups names that are identical after lowercasing/stripping spaces.
        Picks the shortest variant as the canonical name.
        Example: ["Apple Inc", "apple inc"] -> "Apple Inc" (shortest)

    Stage 2 — Fuzzy merge:
        Detects near-duplicate names (e.g. "Jhon" vs "John") using fuzzy
        matching. Clear matches are merged automatically; ambiguous pairs are
        sent to the LLM for a final yes/no decision.

    Args:
        raw_chunk_graph_results: List of ChunkGraphExtractionResult objects
                                 returned by extract_chunk_graph_data_in_parallel.
        provider: Name of the LLM provider to use.

    Returns:
        A dict with key "canonical_name_by_raw_name": a flat map of
        {raw_name -> canonical_name} covering every entity and relationship
        participant seen across all chunks.
    """
    unique_names = set()
    for res in raw_chunk_graph_results:
        for e in res.entities:
            unique_names.add(e.entity_name)
        for r in res.relationships:
            unique_names.add(r.source_entity_name)
            unique_names.add(r.target_entity_name)
            
    if not unique_names:
        return {"canonical_name_by_raw_name": {}}

    grouped_names = group_entities_by_normalized_name(list(unique_names))
    canonical_map = {}
    for _, original_names in grouped_names.items():
        canonical_name = min(original_names, key=len)
        for name in original_names:
            canonical_map[name] = canonical_name
            
    # Then fuzzy merge
    unique_canonicals = list(set(canonical_map.values()))
    fuzzy_candidates = find_fuzzy_merge_candidates(unique_canonicals)
    # Fake scores for now
    same_pairs, ambiguous_pairs, _ = split_clear_cases_from_ambiguous_cases(
        [(l, r, settings.ENTITY_RESOLUTION_DEFAULT_CANDIDATE_SCORE) for l, r in fuzzy_candidates]
    )
    
    for l, r, _ in same_pairs:
        canonical_map[r] = canonical_map.get(l, l)

    client = get_llm(provider)
    decisions = await resolve_ambiguous_entity_pairs(llm_client=client, ambiguous_pairs=ambiguous_pairs)
    
    for decision in decisions:
        if decision.should_merge:
            canonical_map[decision.left_name] = decision.canonical_name
            canonical_map[decision.right_name] = decision.canonical_name

    return {"canonical_name_by_raw_name": canonical_map}


async def persist_document_graph(
    doc: dict[str, Any],
    embeddings: list[list[float]],
    graph_write_payload: dict[str, Any],
) -> None:
    """
    Writes the fully-processed document into Neo4j in three steps:

    1. Document node  — a single :Document node is upserted for the file.
    2. Chunk nodes    — every chunk is saved as a :Chunk node with its
                        embedding vector and the list of entity_ids that
                        appear in that chunk (for fast graph lookups).
    3. Graph layer    — :Entity nodes, :MENTIONED_IN edges, and
                        :RELATES_TO edges are written from graph_write_payload.

    Steps 1-2 open a Neo4j session, save concurrently, then close cleanly
    in the finally block regardless of errors.

    Args:
        doc:                The contextualized document dict (needs "document_id",
                            "source_file", and "chunks" keys).
        embeddings:         Float vectors — one per chunk, same order as doc["chunks"].
        graph_write_payload: Structured payload from build_neo4j_graph_write_payload
                            containing entities, mentions, relationships, and
                            chunk_to_entity_ids mapping.
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
    Builds the JSON response body returned to the API caller after ingestion.

    Reports counts that are useful for monitoring and debugging:
    - raw_entity_count       : total entity mentions found (before deduplication)
    - canonical_entity_count : how many unique real-world entities remain after resolution
    - relationship_count     : number of unique relationships saved to the graph

    Args:
        contextualized_document: The processed document dict (unused here but
                                 kept for a consistent helper signature).
        canonical_graph_payload: CanonicalGraphPersistencePayload with final
                                 .entities and .relationships lists.

    Returns:
        A dict ready to be unpacked into an ingestionResponse model.
    """
    return {
        "status": "success",
        "message": "Document ingested successfully",
        "raw_entity_count": len(canonical_graph_payload.entities),
        "canonical_entity_count": len(set(e["canonical_name"] for e in canonical_graph_payload.entities)),
        "relationship_count": len(canonical_graph_payload.relationships),
    }


async def ingest_document_graph(file_path: str, provider: str = "MISTRAL_AI") -> dict[str, int | str]:
    """
    Orchestrates the full document-to-knowledge-graph pipeline.

    Pipeline stages (in order):
        1. Parse & contextualize  -> build_contextualized_document
        2. Embed chunks           -> build_contextualized_chunk_embeddings
        3. Extract graph (LLM)    -> extract_chunk_graph_data_in_parallel
        4. Resolve entity names   -> resolve_entities_for_graph
        5. Build write payload    -> build_neo4j_graph_write_payload
        6. Persist to Neo4j       -> persist_document_graph

    Args:
        file_path: Absolute path to the document file to ingest.
        provider: LLM provider name ('MISTRAL_AI' or 'DEEPSEEK_AI').

    Returns:
        A summary dict with ingestion stats (entity counts, relationship count).
    """
    log.info("Starting document ingestion pipeline", file_path=file_path)
    
    llm = get_llm(provider)
    doc = await build_contextualized_document(file_path=file_path, client=llm)
    log.info("Document parsed and chunked", num_chunks=len(doc["chunks"]), document_id=doc["document_id"])
    
    embeddings = await build_contextualized_chunk_embeddings(doc["chunks"])
    log.info("Embeddings generated for chunks", num_embeddings=len(embeddings))
    
    raw_results = await extract_chunk_graph_data_in_parallel(doc["chunks"], provider=provider)
    log.info("Graph data extracted from chunks via LLM", num_results=len(raw_results))
    
    resolution_result = await resolve_entities_for_graph(raw_results, provider=provider)
    log.info("Entity resolution complete", num_canonical_names=len(resolution_result["canonical_name_by_raw_name"]))
    
    canonical_payload = rewrite_graph_results_to_canonical_entities(
        raw_results,
        resolution_result["canonical_name_by_raw_name"],
    )
    
    # Send the finished payload to be saved
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
