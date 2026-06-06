from typing import Any, Dict, List
from pydantic import BaseModel

class ExtractedEntity(BaseModel):
    entity_name: str
    entity_type: str
    chunk_id: str
    evidence_text: str

class ExtractedRelationship(BaseModel):
    source_entity_name: str
    relationship_type: str
    target_entity_name: str
    chunk_id: str
    evidence_text: str

class ChunkGraphExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]

class CanonicalGraphPersistencePayload(BaseModel):
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
