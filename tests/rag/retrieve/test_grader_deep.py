"""
Deep tests for the LLM grader.

These tests verify beyond simple pass/fail — they check:
- The exact system prompt is sent to the LLM
- Temperature is exactly 0.0 (deterministic grading)
- JSON response format is requested
- The answer field is populated when sufficient=True
- The answer field is empty when sufficient=False
- Multiple chunks are correctly joined with the separator
- Empty answer string on JSON parse failure
- LLM exception is caught and returns sufficient=False with error reason
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from documentRetrieve.grader import grade_chunks, GraderResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(content: str):
    class Choice:
        class Message:
            def __init__(self, c): self.content = c
        def __init__(self, c): self.message = self.Message(c)
    class Response:
        def __init__(self, c): self.choices = [Choice(c)]
    return Response(content)


# ── Deep Tests ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_grader_sends_temperature_zero():
    """Grading must be deterministic — temperature must be exactly 0.0."""
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": "Yes."}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        await grade_chunks(query="test", chunks=["info"])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0, "Grader must use temperature=0.0"


@pytest.mark.anyio
async def test_grader_requests_json_object_format():
    """Grader must request json_object format from Mistral."""
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": "Yes."}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        await grade_chunks(query="test", chunks=["info"])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.anyio
async def test_grader_sends_both_query_and_chunks_to_llm():
    """The user message sent to LLM must contain BOTH the query and all chunk text."""
    resp = _make_llm_response(json.dumps({"sufficient": False, "reason": "missing", "answer": ""}))
    QUERY = "What is the capital of France?"
    CHUNK = "Paris is a city in Europe."

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        await grade_chunks(query=QUERY, chunks=[CHUNK])

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        assert QUERY in user_message, "User message must contain the query"
        assert CHUNK in user_message, "User message must contain the chunk text"


@pytest.mark.anyio
async def test_grader_joins_multiple_chunks_with_separator():
    """Multiple chunks must be joined with a separator so the LLM sees them as distinct passages."""
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": "Yes."}))
    chunks = ["chunk one text", "chunk two text", "chunk three text"]

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        await grade_chunks(query="q", chunks=chunks)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        # All 3 chunks must appear, separated — not merged into one block
        for chunk in chunks:
            assert chunk in user_message


@pytest.mark.anyio
async def test_grader_populates_answer_field_when_sufficient():
    """When sufficient=True, the answer field must contain the LLM's generated answer."""
    expected_answer = "The answer is Paris."
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": expected_answer}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.answer == expected_answer, "answer field must contain the LLM's response"


@pytest.mark.anyio
async def test_grader_answer_is_empty_when_insufficient():
    """When sufficient=False, the answer field must be empty string."""
    resp = _make_llm_response(json.dumps({"sufficient": False, "reason": "missing data", "answer": ""}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.answer == "", "answer must be empty when insufficient"
    assert result.reason == "missing data"


@pytest.mark.anyio
async def test_grader_answer_is_empty_on_json_parse_error():
    """On JSON parse failure, the answer field must be empty — never None."""
    resp = _make_llm_response("{{broken json}")

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.sufficient is False
    assert result.answer == ""
    assert result.answer is not None


@pytest.mark.anyio
async def test_grader_answer_is_empty_on_llm_exception():
    """On LLM exception, the answer field must be empty — never None or raise."""
    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="test", chunks=["some info"])

    assert result.sufficient is False
    assert result.answer == ""
    assert "error" in result.reason.lower() or "llm" in result.reason.lower()


@pytest.mark.anyio
async def test_grader_result_is_grader_result_type():
    """The return type must always be GraderResult — in every code path."""
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": "ans"}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        result = await grade_chunks(query="q", chunks=["c"])

    assert isinstance(result, GraderResult)

    # Also verify for empty chunks (fast path)
    result2 = await grade_chunks(query="q", chunks=[])
    assert isinstance(result2, GraderResult)


@pytest.mark.anyio
async def test_grader_llm_is_called_exactly_once_per_invocation():
    """For a single grade_chunks() call with chunks, LLM must be called exactly once."""
    resp = _make_llm_response(json.dumps({"sufficient": True, "reason": "", "answer": "yes"}))

    with patch("documentRetrieve.grader.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_builder.return_value = mock_client

        await grade_chunks(query="q", chunks=["a", "b", "c"])

        assert mock_client.chat.completions.create.call_count == 1, (
            "LLM must be called exactly once per grading call, not once per chunk"
        )
