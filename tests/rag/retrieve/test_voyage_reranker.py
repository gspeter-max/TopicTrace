"""
Task 2: Tests for the Voyage AI reranker provider.
All HTTP calls are mocked — no real network traffic.
"""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from topictrace.provider.voyageRerankProvider import rerank_documents


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_voyage_response(documents: list[str], scores: list[float]):
    """Build a fake Voyage /v1/rerank JSON response."""
    return {
        "data": [
            {"index": i, "relevance_score": score}
            for i, score in enumerate(scores)
        ]
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_rerank_returns_top_n_sorted():
    """
    Given 3 documents with scores [0.3, 0.9, 0.6],
    top_k=2 should return the two highest-scored documents in order.
    """
    docs = ["doc_low", "doc_high", "doc_mid"]
    scores = [0.3, 0.9, 0.6]

    fake_resp = MagicMock()
    fake_resp.json.return_value = _make_voyage_response(docs, scores)
    fake_resp.raise_for_status = MagicMock()

    with patch("topictrace.provider.voyageRerankProvider.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = fake_resp
        result = await rerank_documents(query="test query", documents=docs, top_k=2)

    assert result == ["doc_high", "doc_mid"]


@pytest.mark.anyio
async def test_rerank_empty_documents_returns_empty():
    """Passing an empty list must return [] without calling the API."""
    result = await rerank_documents(query="test", documents=[], top_k=5)
    assert result == []


@pytest.mark.anyio
async def test_rerank_respects_top_k():
    """top_k must cap the returned list length."""
    docs = ["a", "b", "c", "d"]
    scores = [0.1, 0.5, 0.8, 0.4]

    fake_resp = MagicMock()
    fake_resp.json.return_value = _make_voyage_response(docs, scores)
    fake_resp.raise_for_status = MagicMock()

    with patch("topictrace.provider.voyageRerankProvider.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = fake_resp
        result = await rerank_documents(query="q", documents=docs, top_k=2)

    assert len(result) == 2
    assert result[0] == "c"   # score 0.8 — highest


@pytest.mark.anyio
async def test_rerank_uses_rerank_endpoint():
    """The provider must call /v1/rerank, not /v1/embeddings."""
    docs = ["x"]
    scores = [0.5]

    fake_resp = MagicMock()
    fake_resp.json.return_value = _make_voyage_response(docs, scores)
    fake_resp.raise_for_status = MagicMock()

    with patch("topictrace.provider.voyageRerankProvider.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = fake_resp
        await rerank_documents(query="q", documents=docs, top_k=1)
        
        called_url = mock_post.call_args[0][0]

    assert "/v1/rerank" in called_url
    assert "/v1/embeddings" not in called_url
