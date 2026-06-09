"""
Deep tests for the LLM grader.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from topictrace.rag.documentRetrieve.grader import GraderResult, grade_chunks

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_llm_response(content: str):
    return AIMessage(content=content)


# ── Deep Tests ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_grader_sends_temperature_zero():
    """Grading must be deterministic — temperature must be exactly 0.0."""
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": "Yes."})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await grade_chunks(query="test", chunks=["info"])

        call_kwargs = mock_llm.bind.call_args[1]
        assert call_kwargs["temperature"] == 0.0, "Grader must use temperature=0.0"


@pytest.mark.anyio
async def test_grader_requests_json_object_format():
    """Grader must request json_object format from Mistral."""
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": "Yes."})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await grade_chunks(query="test", chunks=["info"])

        call_kwargs = mock_llm.bind.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.anyio
async def test_grader_sends_both_query_and_chunks_to_llm():
    """The user message sent to LLM must contain BOTH the query and all chunk text."""
    resp = _make_llm_response(
        json.dumps({"sufficient": False, "reason": "missing", "answer": ""})
    )
    QUERY = "What is the capital of France?"
    CHUNK = "Paris is a city in Europe."

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await grade_chunks(query=QUERY, chunks=[CHUNK])

        messages = mock_bound_llm.ainvoke.call_args[0][0]
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        assert QUERY in user_message, "User message must contain the query"
        assert CHUNK in user_message, "User message must contain the chunk text"


@pytest.mark.anyio
async def test_grader_joins_multiple_chunks_with_separator():
    """Multiple chunks must be joined with a separator so the LLM sees them as distinct passages."""
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": "Yes."})
    )
    chunks = ["chunk one text", "chunk two text", "chunk three text"]

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await grade_chunks(query="q", chunks=chunks)

        messages = mock_bound_llm.ainvoke.call_args[0][0]
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        for chunk in chunks:
            assert chunk in user_message


@pytest.mark.anyio
async def test_grader_populates_answer_field_when_sufficient():
    """When sufficient=True, the answer field must contain the LLM's generated answer."""
    expected_answer = "The answer is Paris."
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": expected_answer})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.answer == expected_answer, (
        "answer field must contain the LLM's response"
    )


@pytest.mark.anyio
async def test_grader_answer_is_empty_when_insufficient():
    """When sufficient=False, the answer field must be empty string."""
    resp = _make_llm_response(
        json.dumps({"sufficient": False, "reason": "missing data", "answer": ""})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.answer == "", "answer must be empty when insufficient"
    assert result.reason == "missing data"


@pytest.mark.anyio
async def test_grader_answer_is_empty_on_json_parse_error():
    """On JSON parse failure, the answer field must be empty — never None."""
    resp = _make_llm_response("{broken json}")

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await grade_chunks(query="test", chunks=["info"])

    assert result.sufficient is False
    assert result.answer == ""
    assert result.answer is not None


@pytest.mark.anyio
async def test_grader_answer_is_empty_on_llm_exception():
    """On LLM exception, the answer field must be empty — never None or raise."""
    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await grade_chunks(query="test", chunks=["some info"])

    assert result.sufficient is False
    assert result.answer == ""
    assert "error" in result.reason.lower() or "llm" in result.reason.lower()


@pytest.mark.anyio
async def test_grader_result_is_grader_result_type():
    """The return type must always be GraderResult — in every code path."""
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": "ans"})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await grade_chunks(query="q", chunks=["c"])

    assert isinstance(result, GraderResult)

    # Also verify for empty chunks (fast path)
    result2 = await grade_chunks(query="q", chunks=[])
    assert isinstance(result2, GraderResult)


@pytest.mark.anyio
async def test_grader_llm_is_called_exactly_once_per_invocation():
    """For a single grade_chunks() call with chunks, LLM must be called exactly once."""
    resp = _make_llm_response(
        json.dumps({"sufficient": True, "reason": "", "answer": "yes"})
    )

    with patch("topictrace.rag.documentRetrieve.grader.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=resp)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await grade_chunks(query="q", chunks=["a", "b", "c"])

        assert mock_bound_llm.ainvoke.call_count == 1, (
            "LLM must be called exactly once per grading call"
        )
