import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.registry import get_tool_definitions, run_tool


def test_get_tool_definitions_returns_three_tools():
    """Test that get_tool_definitions returns definitions for all three tools."""
    tools = get_tool_definitions()
    assert len(tools) == 3

    tool_names = [t["function"]["name"] for t in tools]
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    assert "summarize" in tool_names


def test_get_tool_definitions_have_correct_structure():
    """Test that each tool definition has the required fields."""
    tools = get_tool_definitions()
    for tool in tools:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_run_tool_web_search():
    """Test that run_tool dispatches web_search correctly."""
    session_name = "test-registry-search"
    session_path = create_session(session_name)

    with patch("topictrace.tools.registry.web_search") as mock_search:
        mock_search.return_value = [{"title": "Test", "url": "https://example.com", "snippet": "Test snippet"}]

        result = run_tool("web_search", session_path=session_path, query="test query")

        assert isinstance(result, list)
        assert len(result) == 1
        mock_search.assert_called_once_with(session_path=session_path, query="test query")

    # Cleanup
    shutil.rmtree(session_path)


def test_run_tool_web_fetch():
    """Test that run_tool dispatches web_fetch correctly."""
    session_name = "test-registry-fetch"
    session_path = create_session(session_name)

    with patch("topictrace.tools.registry.web_fetch") as mock_fetch:
        mock_fetch.return_value = "# Test Content"

        result = run_tool("web_fetch", session_path=session_path, url="https://example.com")

        assert isinstance(result, str)
        assert "# Test Content" in result
        mock_fetch.assert_called_once_with(session_path=session_path, url="https://example.com")

    # Cleanup
    shutil.rmtree(session_path)


def test_run_tool_summarize():
    """Test that run_tool dispatches summarize correctly."""
    session_name = "test-registry-summarize"
    session_path = create_session(session_name)

    with patch("topictrace.tools.registry.summarize") as mock_summarize:
        mock_summarize.return_value = "This is a summary."

        result = run_tool("summarize", session_path=session_path, content="test content", query="test query")

        assert isinstance(result, str)
        assert "This is a summary." in result
        mock_summarize.assert_called_once_with(session_path=session_path, content="test content", query="test query")

    # Cleanup
    shutil.rmtree(session_path)


def test_run_tool_unknown_tool():
    """Test that run_tool returns error for unknown tool name."""
    session_name = "test-registry-unknown"
    session_path = create_session(session_name)

    result = run_tool("unknown_tool", session_path=session_path)

    assert isinstance(result, str)
    assert "Error" in result or "error" in result
    assert "unknown_tool" in result

    # Cleanup
    shutil.rmtree(session_path)
