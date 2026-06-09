from typing import Any

from topictrace.db.neo4j import Neo4jClient

# ─── Index Setup ──────────────────────────────────────────────────────────────


async def create_vector_index(
    client: Neo4jClient, index_name: str, dimension: int
) -> None:
    """
    Run once at startup.
    Creates a cosine-similarity vector index on the Chunk.embedding property.
    `dimension` must match the embedding model output
    (e.g. 512 for jina-embeddings-v2-small-en).
    The IF NOT EXISTS guard makes it safe to call on every startup.
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    await client.execute_query(query)


# ─── Write ────────────────────────────────────────────────────────────────────


async def save_chunk(
    client: Neo4jClient,
    chunk: dict[str, Any],
    embedding: list[float],
    entity_ids: list[str] = None,  # This is our new list of codes
) -> None:
    """
    Upserts a single Chunk node with its contextualized text and embedding vector.
    MERGE on chunk_id avoids duplicate nodes when the same document is re-ingested.

    Expected keys in `chunk`:
        chunk_index  - unique identifier for this chunk
        text         - raw chunk text
        context      - LLM-generated contextual prefix
        document_id  - parent document identifier

    Now, it also saves a list of codes (entity_ids) for the important
    names found in this piece of text.
    """
    query = """
    MERGE (c:Chunk {chunk_id: $chunk_id})
    SET c.text        = $text,
        c.context     = $context,
        c.document_id = $document_id,
        c.embedding   = $embedding,
        c.entity_ids  = $entity_ids  // We added this line to save the codes!
    """
    await client.execute_query(
        query,
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "context": chunk["context"],
            "document_id": chunk["document_id"],
            "embedding": embedding,
            "entity_ids": entity_ids or [],  # If there are no codes, save an empty list
        },
    )


ENTITY_WRITE_QUERY = """
UNWIND $entities AS entity_row
MERGE (e:Entity {canonical_name: entity_row.canonical_name})
SET e.entity_type = entity_row.entity_type
"""

MENTION_WRITE_QUERY = """
UNWIND $mentions AS mention_row
MATCH (e:Entity {canonical_name: mention_row.canonical_name})
MATCH (c:Chunk {chunk_id: mention_row.chunk_id})
MERGE (e)-[m:MENTIONED_IN]->(c)
SET m.document_id = mention_row.document_id,
    m.evidence_text = mention_row.evidence_text
"""

RELATIONSHIP_WRITE_QUERY = """
UNWIND $relationships AS relationship_row
MATCH (source:Entity {canonical_name: relationship_row.source_entity_name})
MATCH (target:Entity {canonical_name: relationship_row.target_entity_name})
MERGE (source)-[relationship:RELATES_TO {
    relationship_type: relationship_row.relationship_type,
    document_id: $document_id,
    chunk_id: relationship_row.chunk_id
}]->(target)
SET relationship.evidence_text = relationship_row.evidence_text
"""


async def save_document_node(
    client: Neo4jClient, document_id: str, source_file: str
) -> None:
    """This saves the main document file name into our database so we can link everything back to it."""
    query = """
    MERGE (document:Document {document_id: $document_id})
    SET document.source_file = $source_file
    """
    await client.execute_query(
        query, {"document_id": document_id, "source_file": source_file}
    )


async def save_entity_nodes_and_relationships(
    client: Neo4jClient,
    graph_write_payload: dict[str, Any],
) -> None:
    """This function saves all the best names and how they connect into our graph database, so we can search through them later."""
    await client.execute_query(
        ENTITY_WRITE_QUERY, {"entities": graph_write_payload["entities"]}
    )
    await client.execute_query(
        MENTION_WRITE_QUERY, {"mentions": graph_write_payload["mentions"]}
    )
    await client.execute_query(
        RELATIONSHIP_WRITE_QUERY,
        {
            "relationships": graph_write_payload["relationships"],
            "document_id": graph_write_payload["document_id"],
        },
    )


# ─── Read ─────────────────────────────────────────────────────────────────────


async def retrieve_similar_chunks(
    client: Neo4jClient,
    index_name: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Runs a vector similarity search against `index_name`.

    Returns a list of dicts, each containing:
        full_context  – context prepended to chunk text, ready for LLM prompt
        score         – cosine similarity (0–1, higher = more similar)
        chunk_id      – for traceability
        document_id   – for traceability
    """
    query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN
        node.context + '\n\n' + node.text AS full_context,
        score,
        node.chunk_id    AS chunk_id,
        node.document_id AS document_id,
        node.entity_ids  AS entity_ids
    ORDER BY score DESC
    """
    return await client.execute_query(
        query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "embedding": query_embedding,
        },
    )


async def fetch_entity_neighbors_1hop(
    client: Neo4jClient, entity_ids: list[str]
) -> list[dict[str, Any]]:
    """
    Given a list of entity_ids (extracted from chunks), find all immediate
    (1-hop) relationships for those entities.
    Returns: source canonical_name, relationship_type, target canonical_name, evidence_text.
    """
    if not entity_ids:
        return []

    query = """
    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
    WHERE source.canonical_name IN $entity_ids OR target.canonical_name IN $entity_ids
    RETURN 
        source.canonical_name AS source,
        r.relationship_type AS rel_type,
        target.canonical_name AS target,
        r.evidence_text AS evidence_text
    """
    return await client.execute_query(query, {"entity_ids": entity_ids})
