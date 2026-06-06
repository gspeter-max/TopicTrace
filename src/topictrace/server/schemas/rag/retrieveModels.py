"""
Pydantic models for the retrieval query API.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to be answered.")
    top_k: int = Field(5, description="Number of vector chunks to retrieve initially.")
    top_k_rerank: int = Field(3, description="Number of chunks to keep after reranking.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer to the user's query.")
    intent: str = Field(..., description="The classified intent: 'simple' or 'complex'.")
    used_graph_search: bool = Field(..., description="True if graph traversal was used to answer the question.")
    reason_for_graph_search: str = Field("", description="Why we needed to use the graph database, if we did.")
    context_used: list[str] = Field(..., description="The final reranked text chunks used for context.")
