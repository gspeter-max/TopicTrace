"""
Tests for the retrieve pipeline (LangGraph version).

Strategy: mock `build_rag_graph` at the retrieve module level to return
a pre-configured mock graph. This tests that handle_query correctly:
  - Passes the right initial state to the graph
  - Passes Neo4jClient via config
  - Maps final_state back to QueryResponse
  - Closes the Neo4j client in finally

Deeper per-node logic is tested in test_rag_graph.py.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.models.retrieveModels import QueryRequest, QueryResponse
from app.retrieveAPI import retrieveRouter

# Setup a dummy app to test the router
app = FastAPI()
app.include_router(retrieveRouter)
http_client = TestClient(app)


def _make_mock_graph(final_state: dict) -> MagicMock:
    """Return a mock compiled graph whose ainvoke() returns final_state."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    return mock_graph


# ── handle_query → graph integration ──────────────────────────────────────────

@pytest.mark.anyio
async def test_complex_path_handle_query_returns_correct_response():
    """
    handle_query must map final LangGraph state to QueryResponse correctly.
    Complex path: used_graph_search=True, no reason.
    """
    final_state = {
        "query": "complex query",
        "intent": "complex",
        "answer": "Final Answer",
        "used_graph_search": True,
        "reason_for_graph_search": "",
        "final_context": ["chunk1", "graph facts"],
    }
    mock_graph = _make_mock_graph(final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        req = QueryRequest(query="complex query", top_k=5, top_k_rerank=3)
        res = await __import__("documentRetrieve.retrieve", fromlist=["handle_query"]).handle_query(req)

    assert res.intent == "complex"
    assert res.used_graph_search is True
    assert res.answer == "Final Answer"
    assert res.reason_for_graph_search == ""


@pytest.mark.anyio
async def test_simple_sufficient_path_returns_correct_response():
    """Simple path with grader success: used_graph_search=False."""
    final_state = {
        "query": "simple query",
        "intent": "simple",
        "answer": "Grader Answer",
        "used_graph_search": False,
        "reason_for_graph_search": "",
        "final_context": ["chunk1"],
    }
    mock_graph = _make_mock_graph(final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        req = QueryRequest(query="simple query", top_k=5, top_k_rerank=3)
        from documentRetrieve.retrieve import handle_query
        res = await handle_query(req)

    assert res.intent == "simple"
    assert res.used_graph_search is False
    assert res.answer == "Grader Answer"


@pytest.mark.anyio
async def test_simple_escalation_path_returns_reason():
    """Simple path with grader failure: reason_for_graph_search is set."""
    final_state = {
        "query": "simple query",
        "intent": "simple",
        "answer": "Escalated Answer",
        "used_graph_search": True,
        "reason_for_graph_search": "missing details",
        "final_context": ["chunk1", "graph facts"],
    }
    mock_graph = _make_mock_graph(final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_neo4j.return_value.close = AsyncMock()
        req = QueryRequest(query="simple query", top_k=5, top_k_rerank=3)
        from documentRetrieve.retrieve import handle_query
        res = await handle_query(req)

    assert res.intent == "simple"
    assert res.used_graph_search is True
    assert res.reason_for_graph_search == "missing details"
    assert res.answer == "Escalated Answer"


@pytest.mark.anyio
async def test_neo4j_client_is_closed_even_if_graph_fails():
    """Neo4j client must be closed in finally even when the graph raises."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph exploded"))

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_neo4j.return_value = mock_client

        from documentRetrieve.retrieve import handle_query
        with pytest.raises(RuntimeError, match="graph exploded"):
            await handle_query(QueryRequest(query="q", top_k=5, top_k_rerank=3))

    mock_client.close.assert_awaited_once()


@pytest.mark.anyio
async def test_neo4j_client_passed_via_config():
    """The Neo4j client must be passed as config['configurable']['neo4j_client']."""
    final_state = {
        "intent": "simple", "answer": "ok", "used_graph_search": False,
        "reason_for_graph_search": "", "final_context": [],
    }
    mock_graph = _make_mock_graph(final_state)

    with patch("documentRetrieve.retrieve._rag_graph", mock_graph), \
         patch("documentRetrieve.retrieve.Neo4jClient") as mock_neo4j:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_neo4j.return_value = mock_client

        from documentRetrieve.retrieve import handle_query
        await handle_query(QueryRequest(query="q", top_k=5, top_k_rerank=3))

    call_kwargs = mock_graph.ainvoke.call_args
    config_arg = call_kwargs[1]["config"]
    assert config_arg["configurable"]["neo4j_client"] is mock_client


# ── API endpoint ───────────────────────────────────────────────────────────────

def test_api_route_delegates_to_handle_query():
    """The /retrieve/query endpoint must delegate to handle_query."""
    with patch("app.retrieveAPI.handle_query") as mock_handle:
        mock_handle.return_value = QueryResponse(
            answer="test answer",
            intent="simple",
            used_graph_search=False,
            reason_for_graph_search="",
            context_used=["chunk"]
        )
        response = http_client.post(
            "/retrieve/query",
            json={"query": "test", "top_k": 5, "top_k_rerank": 3}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "test answer"
    assert data["intent"] == "simple"
    assert data["used_graph_search"] is False
