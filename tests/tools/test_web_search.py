import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.web_search import web_search


@patch("topictrace.settings.TAVILY_API_KEY", "test-api-key")
@patch("topictrace.tools.web_search.TavilyClient")
def test_web_search_returns_list_of_results(MockTavilyClient):
    """Test that web_search returns a list of search results."""
    session_name = "test-web-search-results"
    session_path = create_session(session_name)

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

    results = web_search("python tutoring near me", session_path)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Python Tutoring Guide"
    assert results[0]["url"] == "https://example.com/python"
    assert results[0]["snippet"] == "Learn Python basics..."

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.settings.TAVILY_API_KEY", "test-api-key")
@patch("topictrace.tools.web_search.TavilyClient")
def test_web_search_handles_empty_results(MockTavilyClient):
    """Test that web_search handles empty search results gracefully."""
    session_name = "test-web-search-empty"
    session_path = create_session(session_name)

    mock_client = MagicMock()
    MockTavilyClient.return_value = mock_client
    mock_client.search.return_value = {"results": []}

    results = web_search("nonexistent topic xyz", session_path)

    assert isinstance(results, list)
    assert len(results) == 0

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.settings.TAVILY_API_KEY", "test-api-key")
@patch("topictrace.tools.web_search.TavilyClient")
def test_web_search_saves_results_to_file(MockTavilyClient):
    """Test that web_search saves results to search_results.md in session folder."""
    session_name = "test-web-search-save"
    session_path = create_session(session_name)

    mock_client = MagicMock()
    MockTavilyClient.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "content": "Test content snippet"
            }
        ]
    }

    web_search("test query", session_path)

    results_file = os.path.join(session_path, "search_results.md")
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        content = f.read()

    assert "Test Title" in content
    assert "https://example.com" in content

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.settings.TAVILY_API_KEY", "test-api-key")
def test_web_search_uses_cache_when_valid():
    """Test that web_search returns cached results when cache is fresh."""
    session_name = "test-web-search-cache"
    session_path = create_session(session_name)

    # Pre-populate cache
    from topictrace.cache import save_to_cache
    cached_results = [{"title": "Cached Result", "url": "https://cached.com", "snippet": "cached"}]
    save_to_cache(session_path, "search_test-query", cached_results)

    # Should return cached results without calling Tavily
    results = web_search("test query", session_path)

    assert results == cached_results

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.settings.TAVILY_API_KEY", None)
def test_web_search_raises_without_api_key():
    """Test that web_search raises ValueError when TAVILY_API_KEY is not set."""
    session_name = "test-web-search-no-key"
    session_path = create_session(session_name)

    import pytest
    with pytest.raises(ValueError, match="TAVILY_API_KEY not found"):
        web_search("test query", session_path)

    # Cleanup
    shutil.rmtree(session_path)
