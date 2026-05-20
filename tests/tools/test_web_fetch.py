import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.web_fetch import web_fetch


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_returns_markdown_content(mock_get):
    """Test that web_fetch returns clean markdown content from Jina Reader."""
    session_name = "test-web-fetch-content"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.text = "# Python Basics\n\nPython is a programming language..."
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    content = web_fetch("https://example.com/python-basics", session_path)

    assert isinstance(content, str)
    assert "# Python Basics" in content
    mock_get.assert_called_once_with(
        "https://r.jina.ai/https://example.com/python-basics",
        timeout=30
    )

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_handles_request_error(mock_get):
    """Test that web_fetch raises exception when request fails."""
    session_name = "test-web-fetch-error"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    import pytest
    with pytest.raises(Exception, match="Jina Reader returned status 500"):
        web_fetch("https://example.com/broken-page", session_path)

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_saves_content_to_file(mock_get):
    """Test that web_fetch saves fetched content to fetched_pages/ directory."""
    session_name = "test-web-fetch-save"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.text = "# Test Content\n\nSaved to file."
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    web_fetch("https://example.com/page1", session_path)

    # Check that a file was created in fetched_pages/
    fetched_dir = os.path.join(session_path, "fetched_pages")
    files = os.listdir(fetched_dir)
    assert len(files) >= 1

    # Check file content
    with open(os.path.join(fetched_dir, files[0]), "r") as f:
        content = f.read()
    assert "Test Content" in content
    assert "Source: https://example.com/page1" in content

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.tools.web_fetch.requests.get")
def test_web_fetch_uses_cache_when_valid(mock_get):
    """Test that web_fetch returns cached content when cache is fresh."""
    session_name = "test-web-fetch-cache"
    session_path = create_session(session_name)

    # Pre-populate cache
    from topictrace.cache import save_to_cache, create_cache_key
    cached_content = "# Cached Page\n\nThis is cached content."
    cache_key = create_cache_key("fetch", "https://example.com")
    save_to_cache(session_path, cache_key, cached_content)

    # Should return cached content without making HTTP request
    content = web_fetch("https://example.com", session_path)

    assert content == cached_content
    mock_get.assert_not_called()

    # Cleanup
    shutil.rmtree(session_path)


def test_web_fetch_raises_on_empty_url():
    """Test that web_fetch raises ValueError when URL is empty."""
    session_name = "test-web-fetch-empty"
    session_path = create_session(session_name)

    import pytest
    with pytest.raises(ValueError, match="URL cannot be empty"):
        web_fetch("", session_path)

    # Cleanup
    shutil.rmtree(session_path)
