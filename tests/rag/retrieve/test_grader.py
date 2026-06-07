"""
Task 4: Tests for the LLM grader.
All LLM calls are mocked.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from topictrace.rag.documentRetrieve.grader import grade_chunks, GraderResult



# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(content: str):
    class Choice:
        class Message:
            def __init__(self, c):
                self.content = c
        def __init__(self, c):
            self.message = self.Message(c)
            
    class Response:
        def __init__(self, c):
            self.choices = [Choice(c)]
            
    return Response(content)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_grader_returns_sufficient():
    """When the LLM outputs sufficient=True, it returns a GraderResult with sufficient=True."""
    mock_resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "all good"}))

    with patch("topictrace.rag.documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert isinstance(result, GraderResult)
    assert result.sufficient is True
    assert result.reason == "all good"


@pytest.mark.anyio
async def test_grader_returns_insufficient():
    """When the LLM outputs sufficient=False, it returns a GraderResult with sufficient=False."""
    mock_resp = _make_llm_response(json.dumps({"sufficient": False, "reason": "missing info"}))

    with patch("topictrace.rag.documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.sufficient is False
    assert result.reason == "missing info"


@pytest.mark.anyio
async def test_grader_defaults_to_insufficient_on_bad_json():
    """If the LLM returns invalid JSON, fallback to sufficient=False safely."""
    mock_resp = _make_llm_response("not valid json")

    with patch("topictrace.rag.documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.sufficient is False
    assert "error" in result.reason.lower()


@pytest.mark.anyio
async def test_grader_defaults_to_insufficient_on_empty_chunks():
    """If chunks list is empty, return sufficient=False immediately without calling LLM."""
    with patch("topictrace.rag.documentRetrieve.grader.build_mistral_client") as mock_builder:
        result = await grade_chunks(query="test", chunks=[])
        mock_builder.assert_not_called()

    assert result.sufficient is False
    assert "No chunks provided" in result.reason
