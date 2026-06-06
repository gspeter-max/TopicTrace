"""
Retrieval API endpoints.

This file only contains the FastAPI route definitions.
All business logic lives in src/documentRetrieve/retrieve.py.
"""
import structlog
from fastapi import APIRouter

from app.models.retrieveModels import QueryRequest, QueryResponse
from documentRetrieve.retrieve import handle_query

log = structlog.get_logger()

retrieveRouter = APIRouter(prefix="/retrieve", tags=["Retrieval"])


@retrieveRouter.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    POST /retrieve/query

    Accepts a user query, runs the Hybrid Adaptive RAG pipeline,
    and returns an answer with metadata about how it was generated.
    """
    return await handle_query(request)
