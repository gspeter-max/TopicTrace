"""
Pydantic models for the document ingestion API.
"""

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    file_path: str = Field(
        ..., description="Absolute or relative path to the document to ingest."
    )


class IngestionResponse(BaseModel):
    status: str = Field(..., description="'success' or 'error'.")
    message: str = Field(..., description="Human-readable result message.")
    raw_entity_count: int = Field(
        default=0, description="Total entity mentions found before deduplication."
    )
    canonical_entity_count: int = Field(
        default=0, description="Unique real-world entities after resolution."
    )
    relationship_count: int = Field(
        default=0, description="Number of unique relationships saved to the graph."
    )
