"""
Tests for individual LangGraph nodes.

Each node is tested in isolation by:
1. Constructing a RAGState with the inputs that node needs (via _make_state)
2. Calling the node function directly
3. Asserting the returned dict has the correct keys and values

RAGState is a Pydantic BaseModel — nodes use attribute access (state.query),
so tests must pass RAGState instances, not plain dicts.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from topictrace.rag.documentRetrieve.graph.edges import (
    route_after_grader,
    route_after_vector_search,
)
from topictrace.rag.documentRetrieve.graph.nodes import (
    answer_node,
    grade_chunks_node,
    graph_search,
    rerank,
    route_query,
    vector_search,
)
from topictrace.rag.documentRetrieve.graph.state import RAGState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_state(**overrides: Any) -> RAGState:
    """
    Build a RAGState with sensible defaults, overriding only what the test cares about.

    RAGState is a Pydantic BaseModel — every field is required unless given a
    default, so we supply a full set of zero-value defaults here and let callers
    override only the fields relevant to the behaviour under test.
    """
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


def _mock_grader_result(
    sufficient: bool, reason: str = "", answer: str = ""
) -> MagicMock:
    result = MagicMock()
    result.sufficient = sufficient
    result.reason = reason
    result.answer = answer
    return result


# ── route_query ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_route_query_returns_complex_intent() -> None:
    """classify_intent returning 'complex' must produce {"intent": "complex"}."""
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="complex"),
    ):
        result = await route_query("who manages whom?")

    assert result == {"intent": "complex"}
    assert isinstance(result, dict)


@pytest.mark.anyio
async def test_route_query_returns_simple_intent() -> None:
    """classify_intent returning 'simple' must produce {"intent": "simple"}."""
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="simple"),
    ):
        result = await route_query("what is X?")

    assert result == {"intent": "simple"}
    assert isinstance(result, dict)


@pytest.mark.anyio
async def test_route_query_accepts_ragstate_and_extracts_query() -> None:
    """When given a RAGState, route_query must use state.query, not the whole object."""
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="simple"),
    ) as mock_classify:
        await route_query(_make_state(query="test question"))

    mock_classify.assert_awaited_once_with("test question")


# ── vector_search ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_vector_search_returns_chunks_and_texts() -> None:
    """vector_search must return raw_chunks and extracted vector_texts."""
    raw = [{"full_context": "context A", "entity_ids": ["E1"]}]

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            AsyncMock(return_value=raw),
        ),
        patch("topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client"),
    ):
        mock_embed = MagicMock()
        mock_embed.generateEmebedding = AsyncMock(return_value=[0.1, 0.2])
        mock_embed_cls.return_value = mock_embed

        result = await vector_search(_make_state(query="test", top_k=3))

    assert result["raw_chunks"] == raw
    assert result["vector_texts"] == ["context A"]


@pytest.mark.anyio
async def test_vector_search_uses_correct_top_k() -> None:
    """vector_search must forward state.top_k to retrieve_similar_chunks."""
    mock_retrieve = AsyncMock(return_value=[])

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            mock_retrieve,
        ),
        patch("topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client"),
    ):
        mock_embed_cls.return_value.generateEmebedding = AsyncMock(return_value=[0.0])
        await vector_search(_make_state(query="q", top_k=7))

    assert mock_retrieve.call_args[1]["top_k"] == 7


@pytest.mark.anyio
async def test_vector_search_uses_client_from_get_neo4j_client() -> None:
    """vector_search must use the client returned by _get_neo4j_client."""
    sentinel_client = MagicMock()
    mock_retrieve = AsyncMock(return_value=[])

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            mock_retrieve,
        ),
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client",
            return_value=sentinel_client,
        ),
    ):
        mock_embed_cls.return_value.generateEmebedding = AsyncMock(return_value=[0.0])
        await vector_search(_make_state(query="q", top_k=3))

    assert mock_retrieve.call_args[1]["client"] is sentinel_client


# ── grade_chunks_node ─────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_grade_chunks_node_sufficient() -> None:
    """When grader says sufficient=True, node must return grade_sufficient=True with pre-generated answer."""
    mock_result = _mock_grader_result(True, "", "Pre-generated answer")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node(_make_state(query="q", vector_texts=["chunk"]))

    assert result["grade_sufficient"] is True
    assert result["grade_answer"] == "Pre-generated answer"
    assert result["grade_reason"] == ""


@pytest.mark.anyio
async def test_grade_chunks_node_insufficient() -> None:
    """When grader says sufficient=False, node must return grade_sufficient=False with reason."""
    mock_result = _mock_grader_result(False, "missing info", "")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node(_make_state(query="q", vector_texts=["chunk"]))

    assert result["grade_sufficient"] is False
    assert result["grade_reason"] == "missing info"


@pytest.mark.anyio
async def test_grade_chunks_node_only_returns_grade_keys() -> None:
    """grade_chunks_node must return exactly the three grade_* keys."""
    mock_result = _mock_grader_result(True, "", "answer")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node(_make_state(query="q", vector_texts=[]))

    assert set(result.keys()) == {"grade_sufficient", "grade_reason", "grade_answer"}


# ── graph_search ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_graph_search_sets_used_graph_search_true() -> None:
    """graph_search must always set used_graph_search=True."""
    state = _make_state(raw_chunks=[{"entity_ids": ["E1", "E2"]}], grade_reason="")
    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts",
            AsyncMock(return_value="facts"),
        ),
        patch("topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client"),
    ):
        result = await graph_search(state)

    assert result["used_graph_search"] is True
    assert result["graph_facts"] == "facts"


@pytest.mark.anyio
async def test_graph_search_carries_grade_reason_on_escalation() -> None:
    """When escalating from simple path, the grader's reason must be preserved."""
    state = _make_state(raw_chunks=[], grade_reason="missing info")
    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts",
            AsyncMock(return_value=""),
        ),
        patch("topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client"),
    ):
        result = await graph_search(state)

    assert result["reason_for_graph_search"] == "missing info"


@pytest.mark.anyio
async def test_graph_search_extracts_and_deduplicates_entity_ids() -> None:
    """graph_search must pass deduplicated entity_ids from all chunks to gather_graph_facts."""
    state = _make_state(
        raw_chunks=[
            {"entity_ids": ["Alice", "Bob"]},
            {"entity_ids": ["Bob", "Carol"]},
        ],
        grade_reason="",
    )
    mock_gather = AsyncMock(return_value="")
    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts",
            mock_gather,
        ),
        patch("topictrace.rag.documentRetrieve.graph.nodes._get_neo4j_client"),
    ):
        await graph_search(state)

    called_ids = set(mock_gather.call_args[0][1])
    assert called_ids == {"Alice", "Bob", "Carol"}


# ── rerank ────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_rerank_combines_vector_and_graph_facts() -> None:
    """rerank must pass both vector_texts and graph_facts to the reranker."""
    state = _make_state(
        query="q",
        top_k_rerank=3,
        vector_texts=["chunk1", "chunk2"],
        graph_facts="GRAPH KNOWLEDGE:\n- A RELATES_TO B",
    )
    mock_rerank = AsyncMock(
        return_value=["chunk1", "GRAPH KNOWLEDGE:\n- A RELATES_TO B"]
    )
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.rerank_documents", mock_rerank
    ):
        result = await rerank(state)

    docs_passed = mock_rerank.call_args[1]["documents"]
    assert "chunk1" in docs_passed
    assert "GRAPH KNOWLEDGE:\n- A RELATES_TO B" in docs_passed
    assert result["final_context"] == ["chunk1", "GRAPH KNOWLEDGE:\n- A RELATES_TO B"]


@pytest.mark.anyio
async def test_rerank_uses_correct_top_k_rerank() -> None:
    """rerank must forward state.top_k_rerank to rerank_documents."""
    state = _make_state(query="q", top_k_rerank=5, vector_texts=["c1"], graph_facts="")
    mock_rerank = AsyncMock(return_value=["c1"])
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.rerank_documents", mock_rerank
    ):
        await rerank(state)

    assert mock_rerank.call_args[1]["top_k"] == 5


@pytest.mark.anyio
async def test_rerank_skips_empty_graph_facts() -> None:
    """If graph_facts is empty string, it must NOT be added to the rerank document list."""
    state = _make_state(
        query="q",
        top_k_rerank=3,
        vector_texts=["chunk1"],
        graph_facts="",
    )
    mock_rerank = AsyncMock(return_value=["chunk1"])
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.rerank_documents", mock_rerank
    ):
        await rerank(state)

    docs_passed = mock_rerank.call_args[1]["documents"]
    assert "" not in docs_passed
    assert len(docs_passed) == 1


# ── answer_node ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_answer_node_fast_path_uses_grade_answer() -> None:
    """If grade_sufficient=True, answer must come from grade_answer — no LLM call."""
    state = _make_state(
        query="q",
        grade_sufficient=True,
        grade_answer="Pre-generated answer",
        vector_texts=["chunk"],
        final_context=[],
    )
    with patch("topictrace.rag.documentRetrieve.graph.nodes.get_llm") as mock_llm:
        result = await answer_node(state)
        mock_llm.assert_not_called()

    assert result["answer"] == "Pre-generated answer"


@pytest.mark.anyio
async def test_answer_node_standard_path_calls_llm() -> None:
    """If grade_sufficient=False, must call LLM with final_context and return its answer."""
    state = _make_state(
        query="what is X?",
        grade_sufficient=False,
        final_context=["chunk1", "chunk2"],
    )
    mock_resp = AIMessage(content="LLM Answer")
    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=mock_resp)
    mock_llm.bind.return_value = mock_bound_llm

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.get_llm", return_value=mock_llm
        ),
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.get_system_prompt",
            return_value="prompt",
        ),
    ):
        result = await answer_node(state)

    assert result["answer"] == "LLM Answer"


@pytest.mark.anyio
async def test_answer_node_handles_empty_context_gracefully() -> None:
    """If final_context is empty and grade_sufficient=False, return a safe fallback message."""
    state = _make_state(query="q", grade_sufficient=False, final_context=[])
    result = await answer_node(state)

    assert "could not find" in result["answer"].lower()


# ── Conditional edges ─────────────────────────────────────────────────────────


def test_edge_complex_intent_routes_to_graph_search() -> None:
    """Complex queries must bypass grade_chunks and go straight to graph_search."""
    assert route_after_vector_search(_make_state(intent="complex")) == "graph_search"


def test_edge_simple_intent_routes_to_grade_chunks() -> None:
    """Simple queries must go to grade_chunks first."""
    assert route_after_vector_search(_make_state(intent="simple")) == "grade_chunks"


def test_edge_sufficient_grade_routes_to_answer_node() -> None:
    """When grader says sufficient, skip graph_search and go straight to answer."""
    assert route_after_grader(_make_state(grade_sufficient=True)) == "answer_node"


def test_edge_insufficient_grade_routes_to_graph_search() -> None:
    """When grader says not sufficient, escalate to graph_search."""
    assert route_after_grader(_make_state(grade_sufficient=False)) == "graph_search"
