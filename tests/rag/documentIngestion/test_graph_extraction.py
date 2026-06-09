import pydantic
import pytest

from topictrace.rag.documentIngestion.graphRelationshipSchema import (
    ALLOWED_RELATIONSHIP_TYPES,
    get_relationship_schema_prompt_text,
)
from topictrace.rag.documentIngestion.models.graphExtractionModels import (
    ChunkGraphExtractionResult,
    RawGraphEntity,
    RawGraphRelationship,
)


def test_relationship_schema_prompt_lists_every_allowed_relationship():
    """Every allowed relationship type must appear in the LLM system prompt."""
    prompt_text = get_relationship_schema_prompt_text()
    for relationship_type in ALLOWED_RELATIONSHIP_TYPES:
        assert relationship_type in prompt_text


def test_raw_graph_relationship_falls_back_to_related_to_for_unknown_type():
    """Unknown relationship types must silently fall back to RELATED_TO."""
    rel = RawGraphRelationship(
        source_entity_name="Alice",
        relationship_type="WORKS",
        target_entity_name="Neo4j",
        evidence_text="Alice works at Neo4j.",
        chunk_id="resume.pdf::0",
    )
    assert rel.relationship_type == "RELATED_TO"


def test_raw_graph_entity_rejects_empty_evidence_text():
    """Empty evidence_text must raise a validation error."""
    with pytest.raises(ValueError):
        RawGraphEntity(
            entity_name="Alice",
            entity_type="Person",
            chunk_id="doc::0",
            evidence_text="",
        )


def test_chunk_graph_extraction_result_parses_entities_and_relationships():
    """ChunkGraphExtractionResult must parse a valid LLM response payload."""
    result = ChunkGraphExtractionResult(
        **{
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
    )

    assert result.entities[0].entity_name == "Alice"
    assert result.relationships[0].relationship_type == "RELATED_TO"


def test_chunk_graph_extraction_result_raises_on_missing_entity_keys():
    """A payload missing required entity fields must raise a pydantic ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        ChunkGraphExtractionResult(
            **{
                "entities": [{"entity_name": "Alice"}],
                "relationships": [],
            }
        )


def test_chunk_graph_extraction_result_falls_back_to_related_to_for_unknown_type():
    """Unknown relationship type in a full payload must still fall back to RELATED_TO."""
    result = ChunkGraphExtractionResult(
        **{
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
    )
    assert result.relationships[0].relationship_type == "RELATED_TO"


def test_chunk_graph_extraction_result_handles_empty_lists():
    """Empty entities and relationships must parse without error."""
    result = ChunkGraphExtractionResult(**{"entities": [], "relationships": []})
    assert len(result.entities) == 0
    assert len(result.relationships) == 0
