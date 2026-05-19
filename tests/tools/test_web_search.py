from unittest.mock import patch, MagicMock
from topictrace.tools.web_search import web_search


@patch("topictrace.tools.web_search.TavilyClient")
def test_web_search_returns_list_of_results(MockTavilyClient):
    """Test that web_search returns a list of search results."""
    mock_client = MagicMock()
    MockTavilyClient.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Python Tutoring Guide",
                "url": "https://example.com/python",
                "content": "Learn Python basics..."
            }
        ]
    }

    results = web_search("python tutoring near me")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Python Tutoring Guide"
    assert results[0]["url"] == "https://example.com/python"
    assert results[0]["snippet"] == "Learn Python basics..."


@patch("topictrace.tools.web_search.TavilyClient")
def test_web_search_handles_empty_results(MockTavilyClient):
    """Test that web_search handles empty search results gracefully."""
    mock_client = MagicMock()
    MockTavilyClient.return_value = mock_client
    mock_client.search.return_value = {"results": []}

    results = web_search("nonexistent topic xyz")

    assert isinstance(results, list)
    assert len(results) == 0
