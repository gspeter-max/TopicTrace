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
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from documentRetrieve.graph.nodes import answer_node, _extract_entity_ids
from app.models.retrieveModels import QueryRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class MockGraderResult:
    def __init__(self, sufficient, reason, answer=""):
        self.sufficient = sufficient
        self.reason = reason
        self.answer = answer


# ── Tests: _extract_entity_ids() ─────────────────────────────────────────────

def test_extract_entity_ids_deduplicates():
    """Two chunks sharing an entity_id must yield only one copy of that id."""
    chunks = [
        {"entity_ids": ["ent-A", "ent-B"]},
        {"entity_ids": ["ent-B", "ent-C"]},
    ]
    result = _extract_entity_ids(chunks)
    assert sorted(result) == ["ent-A", "ent-B", "ent-C"]


def test_extract_entity_ids_handles_missing_key():
    """Chunks without 'entity_ids' key must not crash the extractor."""
    chunks = [{"full_context": "text only, no entity_ids key"}]
    result = _extract_entity_ids(chunks)
    assert result == []


def test_extract_entity_ids_handles_none_value():
    """Chunks where entity_ids is None must not crash — they must be skipped."""
    chunks = [{"entity_ids": None}, {"entity_ids": ["ent-X"]}]
    result = _extract_entity_ids(chunks)
    assert result == ["ent-X"]


def test_extract_entity_ids_empty_input():
    """Empty chunk list must return empty list."""
    assert _extract_entity_ids([]) == []


def test_extract_entity_ids_empty_ids_list():
    """A chunk with entity_ids=[] must contribute nothing."""
    chunks = [{"entity_ids": []}, {"entity_ids": ["ent-1"]}]
    result = _extract_entity_ids(chunks)
    assert result == ["ent-1"]


# ── Tests: answer_node() — standard path ─────────────────────────────────────

@pytest.mark.anyio
async def test_answer_node_injects_context_into_prompt():
    """The system prompt sent to the LLM must contain the exact context text."""
    state = {
        "query": "query text",
        "grade_sufficient": False,
        "final_context": ["UNIQUE_CONTEXT_BLOCK_12345"],
    }
    llm_resp = _make_llm_response("The answer.")

    with patch("documentRetrieve.graph.nodes.build_mistral_client") as mock_builder, \
         patch("documentRetrieve.graph.nodes.build_final_answer_prompt", side_effect=lambda ctx: f"PROMPT:{ctx}") as mock_prompt:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)
        mock_builder.return_value = mock_client

        await answer_node(state)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")

    assert "UNIQUE_CONTEXT_BLOCK_12345" in system_msg


@pytest.mark.anyio
async def test_answer_node_sends_query_as_user_message():
    """The user query must be sent as the user message."""
    state = {
        "query": "MY_SPECIFIC_QUERY",
        "grade_sufficient": False,
        "final_context": ["some context"],
    }
    llm_resp = _make_llm_response("An answer.")

    with patch("documentRetrieve.graph.nodes.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)
        mock_builder.return_value = mock_client

        await answer_node(state)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

    assert "MY_SPECIFIC_QUERY" in user_msg


@pytest.mark.anyio
async def test_answer_node_calls_llm_exactly_once():
    """answer_node must call the LLM exactly once — not once per chunk."""
    state = {
        "query": "q",
        "grade_sufficient": False,
        "final_context": ["chunk1", "chunk2", "chunk3"],
    }
    llm_resp = _make_llm_response("Answer.")

    with patch("documentRetrieve.graph.nodes.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)
        mock_builder.return_value = mock_client

        await answer_node(state)

        assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.anyio
async def test_answer_node_returns_fallback_on_empty_context():
    """If final_context is empty and not sufficient, must return fallback — never call LLM."""
    state = {"query": "q", "grade_sufficient": False, "final_context": []}

    with patch("documentRetrieve.graph.nodes.build_mistral_client") as mock_builder:
        result = await answer_node(state)
        mock_builder.assert_not_called()

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


@pytest.mark.anyio
async def test_answer_node_returns_fallback_on_llm_exception():
    """If the LLM call raises, must return an error string — never raise or crash."""
    state = {
        "query": "q",
        "grade_sufficient": False,
        "final_context": ["context"],
    }
    with patch("documentRetrieve.graph.nodes.build_mistral_client") as mock_builder:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM is down"))
        mock_builder.return_value = mock_client

        result = await answer_node(state)

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


# ── Tests: handle_query fast path (via mocked graph) ─────────────────────────

@pytest.mark.anyio
async def test_pipeline_skips_final_llm_call_when_grader_is_sufficient():
    """
    On the fast path (grade_sufficient=True), the pipeline must use the
    grader's pre-generated answer directly. answer_node must not call the LLM.
    """
    final_state = {
        "intent": "simple", "answer": "GRADER_GENERATED_ANSWER",
        "used_graph_search": False, "reason_for_graph_search": "",
        "final_context": ["chunk"],
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()

        from documentRetrieve.retrieve import handle_query
        req = QueryRequest(query="simple question", top_k=5, top_k_rerank=3)
        res = await handle_query(req)

    assert res.answer == "GRADER_GENERATED_ANSWER"
    assert res.used_graph_search is False


@pytest.mark.anyio
async def test_pipeline_response_fields_correct_on_fast_path():
    """
    On fast path: used_graph_search=False, reason_for_graph_search='',
    answer comes from grader.
    """
    final_state = {
        "intent": "simple", "answer": "The answer.",
        "used_graph_search": False, "reason_for_graph_search": "",
        "final_context": ["vector_text_here"],
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()

        from documentRetrieve.retrieve import handle_query
        res = await handle_query(QueryRequest(query="q", top_k=5, top_k_rerank=3))

    assert res.used_graph_search is False
    assert res.reason_for_graph_search == ""
    assert res.answer == "The answer."
    assert "vector_text_here" in res.context_used
