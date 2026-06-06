"""
Task 5: Tests for the Graph Agent.
"""
from unittest.mock import AsyncMock, patch

import pytest

from documentRetrieve.graphAgent import gather_graph_facts


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_graph_agent_returns_empty_when_no_entities():
    """If entity_ids is empty, return empty string without calling DB."""
    mock_client = AsyncMock()
    result = await gather_graph_facts(mock_client, [])
    assert result == ""


@pytest.mark.anyio
async def test_graph_agent_uses_1hop_when_successful():
    """If 1-hop returns data, format it properly."""
    mock_client = AsyncMock()
    
    mock_1hop_data = [
        {"source": "Alice", "rel_type": "KNOWS", "target": "Bob", "evidence_text": "Alice knows Bob."}
    ]

    with patch("documentRetrieve.graphAgent.fetch_entity_neighbors_1hop", return_value=mock_1hop_data) as mock_1hop:
        result = await gather_graph_facts(mock_client, ["Alice"])

    mock_1hop.assert_called_once_with(mock_client, ["Alice"])
    
    assert "Alice KNOWS Bob" in result
    assert "Alice knows Bob." in result


@pytest.mark.anyio
async def test_graph_agent_returns_empty_if_1hop_empty():
    """If 1-hop returns empty, it returns an empty string."""
    mock_client = AsyncMock()

    with patch("documentRetrieve.graphAgent.fetch_entity_neighbors_1hop", return_value=[]) as mock_1hop:
        result = await gather_graph_facts(mock_client, ["Alice"])

    mock_1hop.assert_called_once()
    assert result == ""
