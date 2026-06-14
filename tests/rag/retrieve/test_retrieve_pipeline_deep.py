"""
Deep tests for node-level logic: answer generation and entity extraction.

These tests verify:
- The system prompt for answer generation contains the context block
- The grader answer is used directly — no extra LLM call — on the fast path
- answer_node makes exactly ONE LLM call on the standard path
- answer_node returns a fallback string when context is empty
- _extract_entity_ids deduplicates correctly
- _extract_entity_ids handles missing/None entity_ids safely

Functions being tested now live in documentRetrieve.graph.nodes
(moved there as part of LangGraph migration).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from topictrace.rag.documentRetrieve.graph.nodes import (  # pyright: ignore[reportPrivateUsage]
    _extract_entity_ids,
    answer_node,
)
from topictrace.rag.documentRetrieve.graph.state import RAGState
from topictrace.server.schemas.rag.retrieveModels import QueryRequest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_llm_response(content: str) -> AIMessage:
    return AIMessage(content=content)


def _make_state(**overrides: Any) -> RAGState:
    """Build a RAGState with sensible defaults, overriding only what the test needs."""
    defaults: dict[str, Any] = {
        "query": "",
        "top_k": 3,
        "top_k_rerank": 3,
        "intent": "",
        "raw_chunks": [],
        "vector_texts": [],
        "grade_sufficient": False,
        "grade_reason": "",
        "grade_answer": "",
        "graph_facts": "",
        "used_graph_search": False,
        "reason_for_graph_search": "",
        "final_context": [],
        "answer": "",
    }
    defaults.update(overrides)
    return RAGState(**defaults)


def _make_final_state(
    intent: str = "simple",
    answer: str = "Answer",
    used_graph_search: bool = False,
    reason_for_graph_search: str = "",
    final_context: list[str] | None = None,
) -> MagicMock:
    """
    Build a mock LangGraph final-state result.
    handle_query reads fields via final_state.value.<field>.
    """
    value = MagicMock()
    value.intent = intent
    value.answer = answer
    value.used_graph_search = used_graph_search
    value.reason_for_graph_search = reason_for_graph_search
    value.final_context = final_context if final_context is not None else []

    result = MagicMock()
    result.value = value
    return result


def _make_mock_request(graph: MagicMock) -> MagicMock:
    """Build a mock FastAPI Request exposing graph on app.state.ragGraph."""
    mock_request = MagicMock()
    mock_request.app.state.ragGraph = graph
    return mock_request


# ── Tests: _extract_entity_ids() ─────────────────────────────────────────────


def test_extract_entity_ids_deduplicates() -> None:
    """Two chunks sharing an entity_id must yield only one copy of that id."""
    chunks = [
        {"entity_ids": ["ent-A", "ent-B"]},
        {"entity_ids": ["ent-B", "ent-C"]},
    ]
    result = _extract_entity_ids(chunks)
    assert sorted(result) == ["ent-A", "ent-B", "ent-C"]


def test_extract_entity_ids_handles_missing_key() -> None:
    """Chunks without 'entity_ids' key must not crash the extractor."""
    chunks = [{"full_context": "text only, no entity_ids key"}]
    result = _extract_entity_ids(chunks)
    assert result == []


def test_extract_entity_ids_handles_none_value() -> None:
    """Chunks where entity_ids is None must be skipped gracefully."""
    chunks = [{"entity_ids": None}, {"entity_ids": ["ent-X"]}]
    result = _extract_entity_ids(chunks)
    assert result == ["ent-X"]


def test_extract_entity_ids_empty_input() -> None:
    """Empty chunk list must return empty list."""
    assert _extract_entity_ids([]) == []


def test_extract_entity_ids_empty_ids_list() -> None:
    """A chunk with entity_ids=[] must contribute nothing."""
    chunks = [{"entity_ids": []}, {"entity_ids": ["ent-1"]}]
    result = _extract_entity_ids(chunks)
    assert result == ["ent-1"]


# ── Tests: answer_node() — standard path ─────────────────────────────────────


@pytest.mark.anyio
async def test_answer_node_injects_context_into_prompt() -> None:
    """The system prompt sent to the LLM must contain the exact context text."""
    state = _make_state(
        query="query text",
        grade_sufficient=False,
        final_context=["UNIQUE_CONTEXT_BLOCK_12345"],
    )
    llm_resp = _make_llm_response("The answer.")

    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=llm_resp)
    mock_llm.bind.return_value = mock_bound_llm

    def mock_get_system_prompt(prompt_type: Any, input_vars: Any) -> str:
        return f"PROMPT:{input_vars['context_block']}"

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.get_llm", return_value=mock_llm
        ),
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.get_system_prompt",
            side_effect=mock_get_system_prompt,
        ),
    ):
        await answer_node(state)

        messages = mock_bound_llm.ainvoke.call_args[0][0]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")

    assert "UNIQUE_CONTEXT_BLOCK_12345" in system_msg


@pytest.mark.anyio
async def test_answer_node_sends_query_as_user_message() -> None:
    """The user query must be sent as the user-role message."""
    state = _make_state(
        query="MY_SPECIFIC_QUERY",
        grade_sufficient=False,
        final_context=["some context"],
    )
    llm_resp = _make_llm_response("An answer.")

    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=llm_resp)
    mock_llm.bind.return_value = mock_bound_llm

    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.get_llm", return_value=mock_llm
    ):
        await answer_node(state)

        messages = mock_bound_llm.ainvoke.call_args[0][0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

    assert "MY_SPECIFIC_QUERY" in user_msg


@pytest.mark.anyio
async def test_answer_node_calls_llm_exactly_once() -> None:
    """answer_node must call the LLM exactly once — not once per chunk."""
    state = _make_state(
        query="q",
        grade_sufficient=False,
        final_context=["chunk1", "chunk2", "chunk3"],
    )
    llm_resp = _make_llm_response("Answer.")

    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=llm_resp)
    mock_llm.bind.return_value = mock_bound_llm

    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.get_llm", return_value=mock_llm
    ):
        await answer_node(state)

    assert mock_bound_llm.ainvoke.call_count == 1


@pytest.mark.anyio
async def test_answer_node_returns_fallback_on_empty_context() -> None:
    """If final_context is empty and grade_sufficient=False, return fallback without calling LLM."""
    state = _make_state(query="q", grade_sufficient=False, final_context=[])

    with patch("topictrace.rag.documentRetrieve.graph.nodes.get_llm") as mock_get_llm:
        result = await answer_node(state)
        mock_get_llm.assert_not_called()

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


@pytest.mark.anyio
async def test_answer_node_returns_fallback_on_llm_exception() -> None:
    """If the LLM call raises, answer_node must return an error string — never propagate."""
    state = _make_state(
        query="q",
        grade_sufficient=False,
        final_context=["context"],
    )
    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM is down"))
    mock_llm.bind.return_value = mock_bound_llm

    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.get_llm", return_value=mock_llm
    ):
        result = await answer_node(state)

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


# ── Tests: handle_query fast path (via mocked graph) ─────────────────────────


@pytest.mark.anyio
async def test_pipeline_skips_final_llm_call_when_grader_is_sufficient() -> None:
    """
    On the fast path (grade_sufficient=True), the pipeline must use the
    grader's pre-generated answer directly. The graph returns it via .value.answer.
    """
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state(
        intent="simple",
        answer="GRADER_GENERATED_ANSWER",
        used_graph_search=False,
        reason_for_graph_search="",
        final_context=["chunk"],
    )
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        res = await handle_query(
            QueryRequest(query="simple question", top_k=5, top_k_rerank=3),
            mock_request,
        )

    assert res.answer == "GRADER_GENERATED_ANSWER"
    assert res.used_graph_search is False


@pytest.mark.anyio
async def test_pipeline_response_fields_correct_on_fast_path() -> None:
    """
    On fast path: used_graph_search=False, reason_for_graph_search='',
    answer and context_used mapped correctly from final_state.value.
    """
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state(
        intent="simple",
        answer="The answer.",
        used_graph_search=False,
        reason_for_graph_search="",
        final_context=["vector_text_here"],
    )
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        res = await handle_query(
            QueryRequest(query="q", top_k=5, top_k_rerank=3),
            mock_request,
        )

    assert res.used_graph_search is False
    assert res.reason_for_graph_search == ""
    assert res.answer == "The answer."
    assert "vector_text_here" in res.context_used
