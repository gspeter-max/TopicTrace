"""
Deep tests for individual LangGraph nodes.

Each node is tested in isolation by:
1. Constructing a partial RAGState with the inputs that node needs
2. Calling the node function directly
3. Asserting the returned dict has the correct keys and values

This is the main value of LangGraph: each node is a plain async function
that can be unit-tested without running the full graph.
"""

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

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(neo4j_client=None) -> dict:
    """Build a LangGraph-style config with a mocked Neo4j client."""
    return {"configurable": {"neo4j_client": neo4j_client or MagicMock()}}


def _mock_grader_result(sufficient: bool, reason: str = "", answer: str = ""):
    result = MagicMock()
    result.sufficient = sufficient
    result.reason = reason
    result.answer = answer
    return result


# ── route_query ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_route_query_returns_intent():
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="complex"),
    ):
        result = await route_query({"query": "who manages whom?"})
    assert result == {"intent": "complex"}


@pytest.mark.anyio
async def test_route_query_simple_intent():
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="simple"),
    ):
        result = await route_query({"query": "what is X?"})
    assert result == {"intent": "simple"}


@pytest.mark.anyio
async def test_route_query_only_returns_intent_key():
    """Node must return ONLY the keys it is responsible for."""
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.classify_intent",
        AsyncMock(return_value="simple"),
    ):
        result = await route_query({"query": "q"})
    assert set(result.keys()) == {"intent"}


# ── vector_search ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_vector_search_returns_chunks_and_texts():
    raw = [{"full_context": "context A", "entity_ids": ["E1"]}]
    config = _make_config()

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            AsyncMock(return_value=raw),
        ),
    ):
        mock_embed = MagicMock()
        mock_embed.generateEmebedding = AsyncMock(return_value=[0.1, 0.2])
        mock_embed_cls.return_value = mock_embed

        result = await vector_search({"query": "test", "top_k": 3}, config)

    assert result["raw_chunks"] == raw
    assert result["vector_texts"] == ["context A"]


@pytest.mark.anyio
async def test_vector_search_uses_correct_top_k():
    config = _make_config()
    mock_retrieve = AsyncMock(return_value=[])

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            mock_retrieve,
        ),
    ):
        mock_embed_cls.return_value.generateEmebedding = AsyncMock(return_value=[0.0])
        await vector_search({"query": "q", "top_k": 7}, config)

    assert mock_retrieve.call_args[1]["top_k"] == 7


@pytest.mark.anyio
async def test_vector_search_uses_neo4j_client_from_config():
    """vector_search must use the client from config, not create a new one."""
    sentinel_client = MagicMock()
    config = _make_config(sentinel_client)
    mock_retrieve = AsyncMock(return_value=[])

    with (
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.embeddingModel"
        ) as mock_embed_cls,
        patch(
            "topictrace.rag.documentRetrieve.graph.nodes.retrieve_similar_chunks",
            mock_retrieve,
        ),
    ):
        mock_embed_cls.return_value.generateEmebedding = AsyncMock(return_value=[0.0])
        await vector_search({"query": "q", "top_k": 3}, config)

    assert mock_retrieve.call_args[1]["client"] is sentinel_client


# ── grade_chunks_node ─────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_grade_chunks_node_sufficient():
    mock_result = _mock_grader_result(True, "", "Pre-generated answer")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node({"query": "q", "vector_texts": ["chunk"]})

    assert result["grade_sufficient"] is True
    assert result["grade_answer"] == "Pre-generated answer"
    assert result["grade_reason"] == ""


@pytest.mark.anyio
async def test_grade_chunks_node_insufficient():
    mock_result = _mock_grader_result(False, "missing info", "")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node({"query": "q", "vector_texts": ["chunk"]})

    assert result["grade_sufficient"] is False
    assert result["grade_reason"] == "missing info"


@pytest.mark.anyio
async def test_grade_chunks_node_only_returns_grade_keys():
    mock_result = _mock_grader_result(True, "", "answer")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.grade_chunks",
        AsyncMock(return_value=mock_result),
    ):
        result = await grade_chunks_node({"query": "q", "vector_texts": []})
    assert set(result.keys()) == {"grade_sufficient", "grade_reason", "grade_answer"}


# ── graph_search ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_graph_search_sets_used_graph_search_true():
    config = _make_config()
    state = {
        "raw_chunks": [{"entity_ids": ["E1", "E2"]}],
        "grade_reason": "",
    }
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts",
        AsyncMock(return_value="facts"),
    ):
        result = await graph_search(state, config)

    assert result["used_graph_search"] is True
    assert result["graph_facts"] == "facts"


@pytest.mark.anyio
async def test_graph_search_carries_grade_reason_on_escalation():
    """When escalating from simple→graph, the grader's reason must be preserved."""
    config = _make_config()
    state = {"raw_chunks": [], "grade_reason": "missing info"}
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts",
        AsyncMock(return_value=""),
    ):
        result = await graph_search(state, config)
    assert result["reason_for_graph_search"] == "missing info"


@pytest.mark.anyio
async def test_graph_search_extracts_entity_ids_from_chunks():
    config = _make_config()
    state = {
        "raw_chunks": [
            {"entity_ids": ["Alice", "Bob"]},
            {"entity_ids": ["Bob", "Carol"]},
        ],
        "grade_reason": "",
    }
    mock_gather = AsyncMock(return_value="")
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.gather_graph_facts", mock_gather
    ):
        await graph_search(state, config)

    called_ids = set(mock_gather.call_args[0][1])
    assert called_ids == {"Alice", "Bob", "Carol"}  # deduplicated


# ── rerank ────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_rerank_combines_vector_and_graph_facts():
    state = {
        "query": "q",
        "top_k_rerank": 3,
        "vector_texts": ["chunk1", "chunk2"],
        "graph_facts": "GRAPH KNOWLEDGE:\n- A RELATES_TO B",
    }
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
async def test_rerank_uses_correct_top_k_rerank():
    state = {"query": "q", "top_k_rerank": 5, "vector_texts": ["c1"], "graph_facts": ""}
    mock_rerank = AsyncMock(return_value=["c1"])
    with patch(
        "topictrace.rag.documentRetrieve.graph.nodes.rerank_documents", mock_rerank
    ):
        await rerank(state)
    assert mock_rerank.call_args[1]["top_k"] == 5


@pytest.mark.anyio
async def test_rerank_skips_empty_graph_facts():
    """If graph_facts is empty string, it must NOT be added to the rerank list."""
    state = {
        "query": "q",
        "top_k_rerank": 3,
        "vector_texts": ["chunk1"],
        "graph_facts": "",
    }
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
async def test_answer_node_fast_path_uses_grade_answer():
    """If grade_sufficient=True, answer must come from grade_answer — no LLM call."""
    state = {
        "query": "q",
        "grade_sufficient": True,
        "grade_answer": "Pre-generated answer",
        "vector_texts": ["chunk"],
        "final_context": [],
    }
    with patch("topictrace.rag.documentRetrieve.graph.nodes.get_llm") as mock_llm:
        result = await answer_node(state)
        mock_llm.assert_not_called()

    assert result["answer"] == "Pre-generated answer"


@pytest.mark.anyio
async def test_answer_node_standard_path_calls_llm():
    """If grade_sufficient=False, must call LLM with final_context."""
    state = {
        "query": "what is X?",
        "grade_sufficient": False,
        "final_context": ["chunk1", "chunk2"],
    }
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
async def test_answer_node_handles_empty_context_gracefully():
    """If final_context is empty and not sufficient, return a safe fallback."""
    state = {"query": "q", "grade_sufficient": False, "final_context": []}
    result = await answer_node(state)
    assert "could not find" in result["answer"].lower()


# ── Conditional edges ─────────────────────────────────────────────────────────


def test_edge_complex_routes_to_graph_search():
    assert route_after_vector_search({"intent": "complex"}) == "graph_search"


def test_edge_simple_routes_to_grade_chunks():
    assert route_after_vector_search({"intent": "simple"}) == "grade_chunks"


def test_edge_sufficient_routes_to_answer_node():
    assert route_after_grader({"grade_sufficient": True}) == "answer_node"


def test_edge_insufficient_routes_to_graph_search():
    assert route_after_grader({"grade_sufficient": False}) == "graph_search"
