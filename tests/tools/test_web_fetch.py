from unittest.mock import patch, MagicMock
from topictrace.tools.web_fetch import web_fetch


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_returns_markdown_content(mock_get):
    """Test that web_fetch returns clean markdown content from Jina Reader."""
    mock_response = MagicMock()
    mock_response.text = "# Python Basics\n\nPython is a programming language..."
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    content = web_fetch("https://example.com/python-basics")

    assert isinstance(content, str)
    assert "# Python Basics" in content
    mock_get.assert_called_once_with(
        "https://r.jina.ai/https://example.com/python-basics",
        timeout=30
    )


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_handles_request_error(mock_get):
    """Test that web_fetch returns empty string when request fails."""
    mock_get.side_effect = Exception("Network error")

    content = web_fetch("https://example.com/broken-page")

    assert isinstance(content, str)
    assert content == ""
