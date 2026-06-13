"""
Retrieval API endpoints.

This file only contains the FastAPI route definitions.
All business logic lives in src/documentRetrieve/retrieve.py.
"""

from fastapi import APIRouter, Request

from topictrace.rag.documentRetrieve.retrieve import handle_query
from topictrace.server.schemas.rag.retrieveModels import QueryRequest, QueryResponse

retrieveRouter = APIRouter(prefix="/retrieve", tags=["Retrieval"])


@retrieveRouter.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, r:Request) -> QueryResponse:
    """
    POST /retrieve/query

    Accepts a user query, runs the Hybrid Adaptive RAG pipeline,
    and returns an answer with metadata about how it was generated.
    """
    return await handle_query(request, r)


