"""
Task 3: Tests for the LLM-based query intent router.
All LLM calls are mocked.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from topictrace.rag.documentRetrieve.router import classify_intent

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_llm_response(content: str):
    return AIMessage(content=content)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_router_returns_simple_for_simple_query():
    """LLM returns {intent: simple} → classify_intent must return 'simple'."""
    fake_response = _make_llm_response(json.dumps({"intent": "simple"}))

    with patch("topictrace.rag.documentRetrieve.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await classify_intent("What is machine learning?")

    assert result == "simple"


@pytest.mark.anyio
async def test_router_returns_complex_for_complex_query():
    """LLM returns {intent: complex} → classify_intent must return 'complex'."""
    fake_response = _make_llm_response(json.dumps({"intent": "complex"}))

    with patch("topictrace.rag.documentRetrieve.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await classify_intent(
            "How does the AI team relate to the product team?"
        )

    assert result == "complex"


@pytest.mark.anyio
async def test_router_defaults_to_simple_on_bad_json():
    """Malformed JSON from LLM must default to 'simple' — never raise."""
    fake_response = _make_llm_response("not-valid-json")

    with patch("topictrace.rag.documentRetrieve.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await classify_intent("any query")

    assert result == "simple"


@pytest.mark.anyio
async def test_router_defaults_to_simple_on_unexpected_intent():
    """Unknown intent value (e.g., 'medium') must default to 'simple'."""
    fake_response = _make_llm_response(json.dumps({"intent": "medium"}))

    with patch("topictrace.rag.documentRetrieve.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        result = await classify_intent("any query")

    assert result == "simple"


@pytest.mark.anyio
async def test_router_uses_json_object_response_format():
    """Router must request json_object format so Mistral structures its output."""
    fake_response = _make_llm_response(json.dumps({"intent": "simple"}))

    with patch("topictrace.rag.documentRetrieve.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.bind.return_value = mock_bound_llm
        mock_get_llm.return_value = mock_llm

        await classify_intent("hello")

        call_kwargs = mock_llm.bind.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}
        assert call_kwargs.get("temperature") == 0.0
