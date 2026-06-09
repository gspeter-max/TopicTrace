from typing import Any

from pydantic import BaseModel, field_validator

from topictrace.rag.documentIngestion.graphRelationshipSchema import (
    ALLOWED_RELATIONSHIP_TYPES,
)


class ExtractedEntity(BaseModel):
    entity_name: str
    entity_type: str
    chunk_id: str
    evidence_text: str

    @field_validator("evidence_text")
    @classmethod
    def validate_evidence(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("evidence_text cannot be empty")
        return v


class ExtractedRelationship(BaseModel):
    source_entity_name: str
    relationship_type: str
    target_entity_name: str
    chunk_id: str
    evidence_text: str

    @field_validator("relationship_type", mode="before")
    @classmethod
    def validate_rel_type(cls, v: str) -> str:
        if v not in ALLOWED_RELATIONSHIP_TYPES:
            return "RELATED_TO"
        return v


class ChunkGraphExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class CanonicalGraphPersistencePayload(BaseModel):
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


class EntityResolutionDecision(BaseModel):
    left_name: str
    right_name: str
    canonical_name: str


RawGraphEntity = ExtractedEntity
RawGraphRelationship = ExtractedRelationship
