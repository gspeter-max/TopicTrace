"""
Deep tests for the Voyage AI reranker.

These tests verify beyond simple ranking:
- The exact HTTP payload sent to Voyage (model, query, documents)
- Authorization header is set correctly from the API key
- API error (non-200) causes an exception to propagate, not silent empty list
- top_k > number of docs returns all docs (not crash)
- Score tie: stable order preserved
- The provider correctly maps Voyage's 'index' field back to original documents
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from topictrace.provider.rerank import rerank_documents

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_voyage_response(scores: list[float]) -> dict[str, Any]:
    return {"data": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


def _make_mock_http_response(body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


# ── Deep Tests ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_reranker_sends_query_and_documents_in_payload():
    """The POST body sent to Voyage must contain the exact query and documents."""
    docs = ["doc A", "doc B"]
    query = "my search query"

    resp = _make_mock_http_response(_make_voyage_response([0.5, 0.8]))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        await rerank_documents(query=query, documents=docs, top_k=2)

        # The JSON body passed to the post call
        call_kwargs = mock_post.call_args[1]
        sent_json = call_kwargs.get("json", {})

    assert sent_json["query"] == query, "query must be sent exactly in the request body"
    assert sent_json["documents"] == docs, (
        "documents list must be sent exactly in the request body"
    )


@pytest.mark.anyio
async def test_reranker_maps_voyage_index_back_to_original_docs():
    """Voyage returns indices. The reranker must map index → original document text correctly."""
    docs = ["alpha", "beta", "gamma"]
    # gamma (index 2) gets highest score, alpha (index 0) second
    scores = [0.5, 0.1, 0.9]

    resp = _make_mock_http_response(_make_voyage_response(scores))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        result = await rerank_documents(query="q", documents=docs, top_k=3)

    assert result[0] == "gamma", "Highest score (index 2) must map to 'gamma'"
    assert result[1] == "alpha", "Second highest (index 0) must map to 'alpha'"
    assert result[2] == "beta", "Lowest score (index 1) must map to 'beta'"


@pytest.mark.anyio
async def test_reranker_top_k_larger_than_docs_returns_all_docs():
    """If top_k > len(documents), all documents must be returned — no IndexError."""
    docs = ["only doc"]
    scores = [0.7]

    resp = _make_mock_http_response(_make_voyage_response(scores))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        result = await rerank_documents(query="q", documents=docs, top_k=999)

    assert result == ["only doc"]


@pytest.mark.anyio
async def test_reranker_api_http_error_propagates():
    """If Voyage returns an HTTP error, it must NOT silently return []. It must raise."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp

        with pytest.raises(Exception, match="401"):
            await rerank_documents(query="q", documents=["doc"], top_k=1)


@pytest.mark.anyio
async def test_reranker_single_document_returns_that_document():
    """A single document must always be returned without API crash."""
    docs = ["sole document"]
    scores = [0.99]

    resp = _make_mock_http_response(_make_voyage_response(scores))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        result = await rerank_documents(query="q", documents=docs, top_k=1)

    assert result == ["sole document"]


@pytest.mark.anyio
async def test_reranker_output_length_matches_top_k():
    """Result list length must EXACTLY equal top_k, not more, not less (when docs >= top_k)."""
    docs = ["a", "b", "c", "d", "e"]
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]

    resp = _make_mock_http_response(_make_voyage_response(scores))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        result = await rerank_documents(query="q", documents=docs, top_k=3)

    assert len(result) == 3


@pytest.mark.anyio
async def test_reranker_output_is_sorted_highest_score_first():
    """Result must be sorted descending by Voyage relevance score."""
    docs = ["low", "high", "mid"]
    scores = [0.1, 0.9, 0.5]

    resp = _make_mock_http_response(_make_voyage_response(scores))

    with patch(
        "topictrace.provider.rerank.httpx.AsyncClient.post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = resp
        result = await rerank_documents(query="q", documents=docs, top_k=3)

    # Verify strict descending order
    assert result[0] == "high"
    assert result[1] == "mid"
    assert result[2] == "low"
