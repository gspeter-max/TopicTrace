"""
Voyage AI reranker provider.

Calls https://api.voyageai.com/v1/rerank (rerank-2 model),
sorts results by relevance_score descending, returns top_k strings.
Uses httpx.AsyncClient for native async HTTP calls.
"""
import httpx
from topictrace import settings

async def rerank_documents(
    query: str,
    documents: list[str],
    top_k: int = 5,
    api_key: str | None = None,
) -> list[str]:

    """
    Async wrapper around the Voyage rerank API.

    Args:
        query:     The user query to rank documents against.
        documents: List of document strings to rerank.
        top_k:     Maximum number of results to return.
        api_key:   Override key (defaults to settings .voyage_api_key).

    Returns:
        List of document strings sorted by relevance (highest first), capped at top_k.
        Returns [] immediately if documents is empty.
    """
    if not documents:
        return []

    resolved_key = (api_key or settings.RERANKER_CONFIG.VOYAGE_API_KEY or "").strip()
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.RERANKER_CONFIG.VOYAGE_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_k": top_k,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(settings.RERANKER_CONFIG.VOYAGE_RERANK_URL, headers=headers, json=payload)
        response.raise_for_status()

    data = response.json()
    # Each item: {"index": int, "relevance_score": float}
    ranked = sorted(data["data"], key=lambda x: x["relevance_score"], reverse=True)
    return [documents[item["index"]] for item in ranked[:top_k]]
