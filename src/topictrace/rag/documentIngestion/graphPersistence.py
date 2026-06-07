from typing import Any
import hashlib

from topictrace.rag.documentIngestion.models.graphExtractionModels import ChunkGraphExtractionResult, CanonicalGraphPersistencePayload


def generate_stable_entity_id(name: str) -> str:
    """
    This function takes a name (like 'Apple') and turns it into a short 
    12-character code (like 'a1b2c3'). 
    
    It's like giving every important thing a unique ID card so the 
    computer doesn't get confused between different things with the 
    same name.
    """
    # 1. Clean up the name (remove extra spaces and make letters small)
    normalized_name = " ".join(name.lower().strip().split())
    
    # 2. Turn the name into a scrambled secret code (a hash)
    secret_code = hashlib.sha256(normalized_name.encode()).hexdigest()
    
    # 3. Take just the first 12 letters of that code to keep it short
    short_code = secret_code[:12]
    
    return short_code


def rewrite_graph_results_to_canonical_entities(
    raw_chunk_graph_results: list[ChunkGraphExtractionResult],
    canonical_name_by_raw_name: dict[str, str],
) -> CanonicalGraphPersistencePayload:
    """This function takes all the names we found and changes them to the single best name we chose for them, so we can save them correctly."""
    rewritten_entities = []
    rewritten_relationships = []
    for chunk_graph_result in raw_chunk_graph_results:
        for entity in chunk_graph_result.entities:
            rewritten_entities.append({
                "canonical_name": canonical_name_by_raw_name.get(entity.entity_name, entity.entity_name),
                "entity_type": entity.entity_type,
                "chunk_id": entity.chunk_id,
                "evidence_text": entity.evidence_text,
            })
        for relationship in chunk_graph_result.relationships:
            rewritten_relationships.append({
                "source_entity_name": canonical_name_by_raw_name.get(relationship.source_entity_name, relationship.source_entity_name),
                "relationship_type": relationship.relationship_type,
                "target_entity_name": canonical_name_by_raw_name.get(relationship.target_entity_name, relationship.target_entity_name),
                "chunk_id": relationship.chunk_id,
                "evidence_text": relationship.evidence_text,
            })
    return CanonicalGraphPersistencePayload(entities=rewritten_entities, relationships=rewritten_relationships)


def merge_duplicate_relationship_payloads(
    relationship_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """This function takes many identical connections and squishes them into one, making sure we remember all the places we found them."""
    merged_relationships_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relationship_payload in relationship_payloads:
        relationship_key = (
            relationship_payload["source_entity_name"],
            relationship_payload["relationship_type"],
            relationship_payload["target_entity_name"],
        )
        merged_relationships_by_key.setdefault(relationship_key, relationship_payload)
    return list(merged_relationships_by_key.values())


def build_neo4j_graph_write_payload(
    document_id: str,
    canonical_entities: list[dict[str, Any]],
    rewritten_relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    This function gathers all the names and connections we found in a 
    document and prepares them to be saved.
    
    We updated it to also map each piece of text (Chunk) to the unique 
    entity_id of the names found inside it.
    """
    unique_entities = {}
    mentions = []
    # This dictionary will map: piece_of_text -> [list of name entity_ids]
    chunk_to_entity_ids = {}

    for entity in canonical_entities:
        clean_name = entity["canonical_name"]
        # Use our tool from Task 1 to get the code
        entity_id = generate_stable_entity_id(clean_name)

        if clean_name not in unique_entities:
            unique_entities[clean_name] = {
                "canonical_name": clean_name,
                "entity_type": entity["entity_type"],
                "entity_id": entity_id # Remember the code for the entity
            }

        mentions.append({
            "canonical_name": clean_name,
            "entity_id": entity_id,
            "document_id": document_id,
            "chunk_id": entity["chunk_id"],
            "evidence_text": entity["evidence_text"],
        })
        
        # Link this code to the specific piece of text it came from
        parent_chunk_id = entity["chunk_id"]
        if parent_chunk_id not in chunk_to_entity_ids:
            chunk_to_entity_ids[parent_chunk_id] = set()
        chunk_to_entity_ids[parent_chunk_id].add(entity_id)

    # Convert sets to lists so the database can read them easily
    chunk_to_entity_ids = {k: list(v) for k, v in chunk_to_entity_ids.items()}

    for rel in rewritten_relationships:
        rel["document_id"] = document_id
        
    merged_relationships = merge_duplicate_relationship_payloads(rewritten_relationships)

    return {
        "document_id": document_id,
        "entities": list(unique_entities.values()),
        "mentions": mentions,
        "relationships": merged_relationships,
        "chunk_to_entity_ids": chunk_to_entity_ids # Pass our new mapping along
    }
