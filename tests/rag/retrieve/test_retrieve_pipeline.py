"""
Tests for the retrieve pipeline entry point (handle_query).

Strategy:
  - handle_query(userInput, r) now takes a FastAPI Request as second arg.
  - The RAG graph lives on r.app.state.ragGraph (set at startup via lifespan).
  - Tests mock that attribute via a fake Request object — no need to patch
    a module-level symbol that no longer exists.
  - final_state returned by ainvoke is a LangGraph output object whose fields
    are accessed via .value.<field> (not dict keys).

Deeper per-node logic is covered in test_rag_graph.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topictrace.server.routes.rag.retrieveAPI import retrieveRouter
from topictrace.server.schemas.rag.retrieveModels import QueryRequest, QueryResponse

# ── App fixture for API-level tests ───────────────────────────────────────────

app = FastAPI()
app.include_router(retrieveRouter)
http_client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_final_state(
    intent: str = "simple",
    answer: str = "Answer",
    used_graph_search: bool = False,
    reason_for_graph_search: str = "",
    final_context: list[str] | None = None,
) -> MagicMock:
    """
    Build a mock LangGraph final-state result.

    handle_query reads fields via final_state.value.<field> because LangGraph
    wraps the returned state in a result object with a .value attribute.
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
    """
    Build a mock FastAPI Request that exposes the graph on app.state.ragGraph.

    handle_query accesses r.app.state.ragGraph to get the compiled LangGraph.
    """
    mock_request = MagicMock()
    mock_request.app.state.ragGraph = graph
    return mock_request


def _make_mock_graph(final_state: MagicMock) -> MagicMock:
    """Return a mock compiled graph whose ainvoke() returns the given final_state."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    return mock_graph


# ── handle_query integration ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_complex_path_handle_query_returns_correct_response() -> None:
    """
    handle_query must map final LangGraph state fields to QueryResponse correctly.
    Complex path: used_graph_search=True.
    """
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state(
        intent="complex",
        answer="Final Answer",
        used_graph_search=True,
        reason_for_graph_search="",
        final_context=["chunk1", "graph facts"],
    )
    mock_graph = _make_mock_graph(final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        res = await handle_query(
            QueryRequest(query="complex query", top_k=5, top_k_rerank=3),
            mock_request,
        )

    assert res.intent == "complex"
    assert res.used_graph_search is True
    assert res.answer == "Final Answer"
    assert res.reason_for_graph_search == ""


@pytest.mark.anyio
async def test_simple_sufficient_path_returns_correct_response() -> None:
    """Simple path where grader is sufficient: used_graph_search=False."""
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state(
        intent="simple",
        answer="Grader Answer",
        used_graph_search=False,
        reason_for_graph_search="",
        final_context=["chunk1"],
    )
    mock_graph = _make_mock_graph(final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        res = await handle_query(
            QueryRequest(query="simple query", top_k=5, top_k_rerank=3),
            mock_request,
        )

    assert res.intent == "simple"
    assert res.used_graph_search is False
    assert res.answer == "Grader Answer"


@pytest.mark.anyio
async def test_simple_escalation_path_returns_reason() -> None:
    """Simple path where grader escalates: reason_for_graph_search is set."""
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state(
        intent="simple",
        answer="Escalated Answer",
        used_graph_search=True,
        reason_for_graph_search="missing details",
        final_context=["chunk1", "graph facts"],
    )
    mock_graph = _make_mock_graph(final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        res = await handle_query(
            QueryRequest(query="simple query", top_k=5, top_k_rerank=3),
            mock_request,
        )

    assert res.intent == "simple"
    assert res.used_graph_search is True
    assert res.reason_for_graph_search == "missing details"
    assert res.answer == "Escalated Answer"


@pytest.mark.anyio
async def test_neo4j_client_is_closed_even_if_graph_raises() -> None:
    """Neo4j client must be closed in the finally block even when the graph raises."""
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    exploding_graph = MagicMock()
    exploding_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph exploded"))
    mock_request = _make_mock_request(exploding_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_neo4j.return_value = mock_client

        with pytest.raises(RuntimeError, match="graph exploded"):
            await handle_query(
                QueryRequest(query="q", top_k=5, top_k_rerank=3),
                mock_request,
            )

    mock_client.close.assert_awaited_once()


@pytest.mark.anyio
async def test_neo4j_client_passed_via_config_to_ainvoke() -> None:
    """handle_query must pass the Neo4j client inside config['configurable']['neo4j_client']."""
    from topictrace.rag.documentRetrieve.retrieve import handle_query

    final_state = _make_final_state()
    mock_graph = _make_mock_graph(final_state)
    mock_request = _make_mock_request(mock_graph)

    with patch("topictrace.rag.documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_neo4j.return_value = mock_client

        await handle_query(
            QueryRequest(query="q", top_k=5, top_k_rerank=3),
            mock_request,
        )

    call_kwargs = mock_graph.ainvoke.call_args[1]
    assert call_kwargs["config"]["configurable"]["neo4j_client"] is mock_client


# ── API endpoint ───────────────────────────────────────────────────────────────


def test_api_route_delegates_to_handle_query() -> None:
    """The /retrieve/query endpoint must call handle_query and return its result."""
    with patch("topictrace.server.routes.rag.retrieveAPI.handle_query") as mock_handle:
        mock_handle.return_value = QueryResponse(
            answer="test answer",
            intent="simple",
            used_graph_search=False,
            reason_for_graph_search="",
            context_used=["chunk"],
        )
        response = http_client.post(
            "/retrieve/query", json={"query": "test", "top_k": 5, "top_k_rerank": 3}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "test answer"
    assert data["intent"] == "simple"
    assert data["used_graph_search"] is False
