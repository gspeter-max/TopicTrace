"""
Ingestion API endpoints.

This file only contains the FastAPI route definitions.
All business logic lives in src/documentIngestion/ingestion.py.
"""

from fastapi import APIRouter

from topictrace import log
from topictrace.rag.documentIngestion.ingestion import ingest_document_graph
from topictrace.server.schemas.rag.ingestionModels import (
    IngestionRequest,
    IngestionResponse,
)

ingestionRouter = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@ingestionRouter.post("/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest) -> IngestionResponse:
    """
    POST /ingestion/ingest

    Accepts a file path, 
    file_path: str = Field(
        ..., description="Absolute or relative path to the document to ingest."
    )
    runs the full document-to-knowledge-graph pipeline,
    and returns a summary of what was ingested.
    """
    try:
        summary = await ingest_document_graph(request.file_path)
        return IngestionResponse(**summary)
    except Exception as e:
        log.error("Ingestion failed", error=str(e), exc_info=True)
        return IngestionResponse(status="error", message=str(e))
