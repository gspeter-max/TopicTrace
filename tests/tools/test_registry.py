from unittest.mock import patch, MagicMock
from topictrace.tools.registry import run_tool


def test_run_tool_web_search():
    """Test that run_tool dispatches web_search correctly."""
    with patch("topictrace.tools.registry.web_search") as mock_search:
        mock_search.return_value = [{"title": "Test", "url": "https://example.com", "snippet": "Test snippet"}]

        result = run_tool("web_search", query="test query")

        assert isinstance(result, list)
        assert len(result) == 1
        mock_search.assert_called_once_with(query="test query")


def test_run_tool_web_fetch():
    """Test that run_tool dispatches web_fetch correctly."""
    with patch("topictrace.tools.registry.web_fetch") as mock_fetch:
        mock_fetch.return_value = "# Test Content"

        result = run_tool("web_fetch", url="https://example.com")

        assert isinstance(result, str)
        assert "# Test Content" in result
        mock_fetch.assert_called_once_with(url="https://example.com")


def test_run_tool_summarize():
    """Test that run_tool dispatches summarize correctly."""
    with patch("topictrace.tools.registry.summarize") as mock_summarize:
        mock_summarize.return_value = "This is a summary."

        result = run_tool("summarize", content="test content", query="test query")

        assert isinstance(result, str)
        assert "This is a summary." in result
        mock_summarize.assert_called_once_with(content="test content", query="test query")


def test_run_tool_unknown_tool():
    """Test that run_tool returns error for unknown tool name."""
    result = run_tool("unknown_tool")

    assert isinstance(result, str)
    assert "Error" in result or "error" in result
    assert "unknown_tool" in result
