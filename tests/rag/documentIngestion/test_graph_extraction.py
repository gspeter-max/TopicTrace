import pytest
import pydantic

from src.documentIngestion.graphRelationshipSchema import (
    ALLOWED_RELATIONSHIP_TYPES,
    get_relationship_schema_prompt_text,
)
from src.documentIngestion.models.graphExtractionModels import RawGraphRelationship, RawGraphEntity


def test_relationship_schema_prompt_lists_every_allowed_relationship():
    """This test checks to make sure every connection type we allow is written down in the instructions we give to the AI."""
    prompt_text = get_relationship_schema_prompt_text()

    for relationship_type in ALLOWED_RELATIONSHIP_TYPES:
        assert relationship_type in prompt_text


def test_raw_graph_relationship_falls_back_to_related_to_for_unknown_type():
    """This test makes sure that if the AI tries to use a connection type we did not allow, we fall back to RELATED_TO instead of erroring."""
    rel = RawGraphRelationship(
        source_entity_name="Alice",
        relationship_type="WORKS",
        target_entity_name="Neo4j",
        evidence_text="Alice works at Neo4j.",
        chunk_id="resume.pdf::0",
    )
    assert rel.relationship_type == "RELATED_TO"

def test_raw_graph_entity_rejects_empty_evidence_text():
    """This edge-case test makes sure our system properly catches errors if the AI gives us an empty string for evidence."""
    with pytest.raises(ValueError):
        RawGraphEntity(
            entity_name="Alice",
            entity_type="Person",
            chunk_id="doc::0",
            evidence_text="",
        )

from src.documentIngestion.graphExtraction import parse_chunk_graph_extraction_response


def test_parse_chunk_graph_extraction_response_returns_entities_and_relationships():
    """This test checks if we can correctly understand the answer the AI gives us and pull out the names and connections from it."""
    response_payload = {
        "entities": [
            {
                "entity_name": "Alice",
                "entity_type": "Person",
                "chunk_id": "resume.pdf::0",
                "evidence_text": "Alice works at Neo4j.",
            }
        ],
        "relationships": [
            {
                "source_entity_name": "Alice",
                "relationship_type": "RELATED_TO",
                "target_entity_name": "Neo4j",
                "evidence_text": "Alice works at Neo4j.",
                "chunk_id": "resume.pdf::0",
            }
        ],
    }

    result = parse_chunk_graph_extraction_response(response_payload)

    assert result.entities[0].entity_name == "Alice"
    assert result.relationships[0].relationship_type == "RELATED_TO"

def test_parse_chunk_graph_extraction_handles_missing_keys():
    """This test makes sure that if the AI gives us a bad dictionary missing keys, we throw a validation error."""
    response_payload = {
        "entities": [{"entity_name": "Alice"}],
        "relationships": []
    }
    with pytest.raises(pydantic.ValidationError):
        parse_chunk_graph_extraction_response(response_payload)

def test_parse_chunk_graph_extraction_response_falls_back_to_related_to_for_unknown_type():
    response_payload = {
        "entities": [],
        "relationships": [
            {
                "source_entity_name": "Alice",
                "relationship_type": "SENDS_TO",
                "target_entity_name": "Neo4j",
                "evidence_text": "Alice sends data to Neo4j.",
                "chunk_id": "resume.pdf::0",
            }
        ],
    }

    result = parse_chunk_graph_extraction_response(response_payload)
    assert result.relationships[0].relationship_type == "RELATED_TO"

def test_parse_chunk_graph_extraction_handles_empty_lists():
    """This test makes sure we handle empty lists gracefully when nothing is found."""
    response_payload = {
        "entities": [],
        "relationships": []
    }
    result = parse_chunk_graph_extraction_response(response_payload)
    assert len(result.entities) == 0
    assert len(result.relationships) == 0
