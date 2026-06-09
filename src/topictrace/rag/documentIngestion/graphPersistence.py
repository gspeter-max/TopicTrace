import hashlib
from collections import defaultdict
from typing import Any

from topictrace import log
from topictrace.rag.documentIngestion.models.graphExtractionModels import (
    CanonicalGraphPersistencePayload,
    ChunkGraphExtractionResult,
)


def generate_stable_entity_id(name: str) -> str:
    """Converts an entity canonical name into a stable 12-character hex ID."""

    if not name or not name.strip():
        raise ValueError("Entity name must not be empty when generating a stable ID.")

    normalized_name = " ".join(name.lower().strip().split())
    return hashlib.sha256(normalized_name.encode()).hexdigest()[:12]


def rewrite_graph_results_to_canonical_entities(
    raw_chunk_graph_results: list[ChunkGraphExtractionResult],
    canonical_name_by_raw_name: dict[str, list[str]],
    canonical_by_alias: dict[str, str] | None = None,
) -> CanonicalGraphPersistencePayload:
    """
    Rewrites every raw entity and relationship name to its resolved canonical name.

    If an entity name is determined to be an alias of another canonical entity, it is
    skipped from the entities list (since the canonical entity already tracks all of its
    aliases in the database). Unmapped entities are preserved as-is.

    For relationships, both the source and target entity names are resolved to their
    respective canonical names. Duplicate relationships within the same document are
    filtered out to keep the output dataset clean.

    Args:
        raw_chunk_graph_results: A list of raw graph extraction results from the chunks.
        canonical_name_by_raw_name: A dictionary mapping canonical names to their lists of aliases.
        canonical_by_alias: A flat lookup dictionary mapping raw/alias names to canonical names.
                            If None, it is constructed dynamically for backward compatibility.

    Returns:
        A CanonicalGraphPersistencePayload container holding the lists of canonicalized
        entities and relationships.
    """
    if not isinstance(canonical_name_by_raw_name, dict):
        raise TypeError(
            f"canonical_name_by_raw_name must be a dict, got {type(canonical_name_by_raw_name)!r}"
        )

    # Build the alias-to-canonical lookup dictionary if it was not pre-computed
    if canonical_by_alias is None:
        canonical_by_alias = {}
        for canonical, aliases in canonical_name_by_raw_name.items():
            canonical_by_alias[canonical] = canonical
            for alias in aliases:
                canonical_by_alias[alias] = canonical

    rewritten_entities: list[dict[str, Any]] = []
    rewritten_relationships: list[dict[str, Any]] = []
    seen_relationships: set[tuple[str, str, str]] = set()

    for chunk_entitys_relationships in raw_chunk_graph_results:
        # 1. Process entities and filter out aliases
        for entity in chunk_entitys_relationships.entities:
            entity_name = entity.entity_name
            if canonical_name_by_raw_name and entity_name not in canonical_name_by_raw_name:
                if entity_name in canonical_by_alias:
                    continue

            rewritten_entities.append(
                {
                    "canonical_name": canonical_by_alias.get(entity_name, entity_name),
                    "entity_type": entity.entity_type,
                    "chunk_id": entity.chunk_id,
                    "evidence_text": entity.evidence_text,
                    "alias": canonical_name_by_raw_name.get(entity_name, [])
                }
            )

        # 2. Process relationships and canonicalize endpoints
        for relationship in chunk_entitys_relationships.relationships:
            # Fall back to original name if the endpoint is not present in our lookup map
            source_canonical = canonical_by_alias.get(relationship.source_entity_name, relationship.source_entity_name)
            target_canonical = canonical_by_alias.get(relationship.target_entity_name, relationship.target_entity_name)

            if canonical_name_by_raw_name:
                if (
                    source_canonical not in canonical_name_by_raw_name or \
                    target_canonical not in canonical_name_by_raw_name
                ):
                    continue 

            # Deduplicate relationships by checking the (source, type, target) triple
            relationship_key = (source_canonical, relationship.relationship_type, target_canonical)
            if relationship_key in seen_relationships:
                continue
            seen_relationships.add(relationship_key)

            rewritten_relationships.append(
                {
                    "source_entity_name": source_canonical,
                    "relationship_type": relationship.relationship_type,
                    "target_entity_name": target_canonical,
                    "chunk_id": relationship.chunk_id,
                    "evidence_text": relationship.evidence_text,
                }
            )

    log.debug(
        "Rewrote %d entities and %d relationships to canonical names.",
        len(rewritten_entities),
        len(rewritten_relationships),
    )
    return CanonicalGraphPersistencePayload(
        entities=rewritten_entities, relationships=rewritten_relationships
    )

def build_neo4j_graph_write_payload(
    document_id: str,
    canonical_entities: list[dict[str, Any]],
    rewritten_relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assembles the final structured payload ready to be written to Neo4j.

    Responsibilities:
    - Deduplicates entities by canonical name (first occurrence wins for type).
    - Builds a flat mention list (one row per entity-chunk occurrence).
    - Computes a stable entity_id for every unique canonical name.
    - Produces a chunk_to_entity_ids mapping used to annotate Chunk nodes.
    - Deduplicates relationships.

    Args:
        document_id: Unique identifier for the source document.
        canonical_entities: Rewritten entity dicts from
                            rewrite_graph_results_to_canonical_entities.
        rewritten_relationships: Rewritten relationship dicts from the same fn.

    Returns:
        Dict with keys: document_id, entities, mentions, relationships,
        chunk_to_entity_ids.

    Raises:
        ValueError: If document_id is empty.
    """
    if not document_id or not document_id.strip():
        raise ValueError(
            "document_id must not be empty when building the graph payload."
        )

    unique_entities: dict[str, dict[str, Any]] = {}
    mentions: list[dict[str, Any]] = []
    chunk_to_entity_ids: defaultdict[str, list[str]] = defaultdict(list)

    for entity in canonical_entities:
        clean_name: str = entity["canonical_name"]

        if not clean_name.strip():
            log.warning(
                "Skipping entity with empty canonical_name in document %r.", document_id
            )
            continue

        entity_id = generate_stable_entity_id(clean_name).strip() 

        if clean_name not in unique_entities:
            unique_entities[clean_name] = {
                "canonical_name": clean_name,
                "entity_type": entity["entity_type"],
                "entity_id": entity_id,
            }

        mentions.append(
            {
                "canonical_name": clean_name,
                "entity_id": entity_id,
                "document_id": document_id,
                "chunk_id": entity["chunk_id"],
                "evidence_text": entity["evidence_text"],
            }
        )
        if entity_id not in chunk_to_entity_ids[entity["chunk_id"]]:
            chunk_to_entity_ids[entity["chunk_id"]].append(entity_id)

    log.debug(
        "Built graph payload for document %r: %d unique entities, %d mentions, %d relationships.",
        document_id,
        len(unique_entities),
        len(mentions),
        len(rewritten_relationships),
    )

    return {
        "document_id": document_id,
        "entities": list(unique_entities.values()),
        "mentions": mentions,
        "relationships": rewritten_relationships,
        "chunk_to_entity_ids": chunk_to_entity_ids,
    }
