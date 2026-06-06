from src.documentIngestion.graphPersistence import rewrite_graph_results_to_canonical_entities
from src.documentIngestion.models.graphExtractionModels import ChunkGraphExtractionResult, RawGraphEntity, RawGraphRelationship


def test_rewrite_graph_results_to_canonical_entities_updates_relationship_endpoints():
    """This test checks that when we change a name to its best version, 
    we also update all the connections that use that name to point to the new best version."""
    raw_results = [
        ChunkGraphExtractionResult(
            entities=[RawGraphEntity(entity_name="Mr Musk", entity_type="Person", chunk_id="doc::0", evidence_text="Mr Musk spoke.")],
            relationships=[RawGraphRelationship(source_entity_name="Mr Musk", relationship_type="RELATED_TO", target_entity_name="Tesla", evidence_text="Mr Musk leads Tesla.", chunk_id="doc::0")],
        )
    ]
    canonical_name_map = {"Mr Musk": "Elon Musk", "Tesla": "Tesla"}

    rewritten = rewrite_graph_results_to_canonical_entities(raw_results, canonical_name_map)

    assert rewritten.relationships[0]["source_entity_name"] == "Elon Musk"

def test_rewrite_graph_results_to_canonical_entities_handles_missing_mappings():
    """Edge case: Make sure it gracefully handles missing mappings by keeping the raw name."""
    raw_results = [
        ChunkGraphExtractionResult(
            entities=[RawGraphEntity(entity_name="Unknown Person", entity_type="Person", chunk_id="doc::0", evidence_text="Unknown Person spoke.")],
            relationships=[],
        )
    ]
    canonical_name_map = {}
    rewritten = rewrite_graph_results_to_canonical_entities(raw_results, canonical_name_map)

    assert rewritten.entities[0]["canonical_name"] == "Unknown Person"


def test_build_neo4j_graph_write_payload_contains_mentions_and_relationships():
    """This test makes sure that the package of data we send to the database contains all the names, where they were mentioned, and how they connect."""
    from src.documentIngestion.graphPersistence import build_neo4j_graph_write_payload
    payload = build_neo4j_graph_write_payload(
        document_id="resume.pdf",
        canonical_entities=[{"canonical_name": "Elon Musk", "entity_type": "Person", "chunk_id": "doc::0", "evidence_text": "Elon spoke"}],
        rewritten_relationships=[{"source_entity_name": "Elon Musk", "relationship_type": "RELATED_TO", "target_entity_name": "Tesla", "evidence_text": "Elon leads Tesla", "chunk_id": "doc::0"}],
    )

    assert payload["document_id"] == "resume.pdf"
    assert payload["relationships"][0]["relationship_type"] == "RELATED_TO"
    assert payload["mentions"][0]["chunk_id"] == "doc::0"

def test_build_neo4j_graph_write_payload_deduplicates_entities():
    """Edge case: ensure entities are deduplicated by canonical name in the payload."""
    from src.documentIngestion.graphPersistence import build_neo4j_graph_write_payload
    payload = build_neo4j_graph_write_payload(
        document_id="doc.pdf",
        canonical_entities=[
            {"canonical_name": "A", "entity_type": "Person", "chunk_id": "1", "evidence_text": "e1"},
            {"canonical_name": "A", "entity_type": "Person", "chunk_id": "2", "evidence_text": "e2"}
        ],
        rewritten_relationships=[]
    )
    assert len(payload["entities"]) == 1
    assert len(payload["mentions"]) == 2
