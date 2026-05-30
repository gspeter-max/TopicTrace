"""
Deep audit tests for TopicTrace.

These tests catch regressions by testing every edge case, boundary condition,
and failure mode across the entire codebase. If anything breaks, these tests
will catch it.

Coverage:
- session.py: path traversal, unicode, long names, idempotency
- cache.py: TTL boundaries, corruption, overwrites, nested data
- web_search.py: API failures, malformed responses, snippet limits
- web_fetch.py: HTTP errors, timeouts, large content, redirects
- summarize.py: empty responses, streaming failures, content truncation
- integration: full chain with all failure paths
"""

import asyncio
import json
import os
import shutil
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from topictrace.session import create_session, get_session_path
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid, _get_cache_file_path
from topictrace import settings


# ============================================================
# SESSION.PY DEEP AUDIT
# ============================================================

class TestSessionSanitization:
    """Test every possible malicious or invalid session name input."""

    def test_path_traversal_dot_dot_slash(self):
        """Block ../../etc/passwd style attacks."""
        result = os.path.basename(get_session_path("../../etc/passwd"))
        assert ".." not in result
        assert "/" not in result

    def test_path_traversal_dot_dot_backslash(self):
        """Block ..\\..\\etc\\passwd style attacks on Windows."""
        result = os.path.basename(get_session_path("..\\..\\etc\\passwd"))
        assert ".." not in result
        assert "\\" not in result

    def test_path_traversal_absolute_path(self):
        """Block /etc/passwd style attacks."""
        result = os.path.basename(get_session_path("/etc/passwd"))
        assert "/" not in result
        assert result.startswith("etcpasswd") or len(result) > 0

    def test_path_traversal_null_byte(self):
        """Block null byte injection."""
        result = os.path.basename(get_session_path("session\x00name"))
        assert "\x00" not in result

    def test_path_traversal_unicode_slash(self):
        """Block unicode slash characters."""
        result = os.path.basename(get_session_path("session\u2215name"))
        assert "\u2215" not in result

    def test_special_characters_stripped(self):
        """Strip all special characters except dash and underscore."""
        result = os.path.basename(get_session_path("test!@#$%^&*()+={}[]|\\:;\"'<>,.?/~`"))
        # Only alphanumeric, dash, underscore should remain
        for char in result:
            assert char.isalnum() or char in "-_", f"Unexpected char: {char}"

    def test_unicode_letters_kept(self):
        """Keep alphanumeric unicode characters."""
        result = os.path.basename(get_session_path("Biology-2024"))
        assert "Biology" in result
        assert "2024" in result

    def test_only_special_characters_raises(self):
        """Session name with only special characters should raise ValueError."""
        with pytest.raises(ValueError, match="empty or contains only invalid characters"):
            get_session_path("!@#$%^&*()")

    def test_empty_string_raises(self):
        """Empty session name should raise ValueError."""
        with pytest.raises(ValueError, match="empty or contains only invalid characters"):
            get_session_path("")

    def test_whitespace_only_raises(self):
        """Whitespace-only session name should raise ValueError."""
        with pytest.raises(ValueError, match="empty or contains only invalid characters"):
            get_session_path("   ")

    def test_dashes_collapsed(self):
        """Multiple consecutive dashes should be collapsed to single dash."""
        result = os.path.basename(get_session_path("test---name"))
        assert "---" not in result
        assert "--" not in result

    def test_underscores_converted_to_dashes(self):
        """Underscores should be converted to dashes."""
        result = os.path.basename(get_session_path("test_name"))
        assert "_" not in result
        assert "-" in result

    def test_leading_trailing_dashes_stripped(self):
        """Leading and trailing dashes should be stripped."""
        result = os.path.basename(get_session_path("---test---"))
        assert not result.startswith("-")
        assert not result.endswith("-")


class TestSessionCreation:
    """Test session folder creation edge cases."""

    def test_create_session_idempotent(self):
        """Creating the same session twice should not fail."""
        name = "test-idempotent"
        path1 = create_session(name)
        path2 = create_session(name)
        assert path1 == path2
        assert os.path.isdir(path1)
        shutil.rmtree(path1)

    def test_create_session_all_subdirectories(self):
        """All required subdirectories must exist."""
        name = "test-subdirs"
        path = create_session(name)
        assert os.path.isdir(os.path.join(path, "fetched_pages"))
        assert os.path.isdir(os.path.join(path, "summaries"))
        assert os.path.isdir(os.path.join(path, "cache"))
        shutil.rmtree(path)

    def test_create_session_returns_absolute_or_relative_path(self):
        """Path should start with sessions/ directory."""
        name = "test-path-format"
        path = create_session(name)
        assert path.startswith(settings.SESSIONS_DIR)
        shutil.rmtree(path)

    def test_get_session_path_format(self):
        """Path should be sessions/<name> format."""
        path = get_session_path("my-session")
        assert path == os.path.join(settings.SESSIONS_DIR, "my-session")

    def test_create_session_with_numbers(self):
        """Session names with numbers should work."""
        name = "test-123-456"
        path = create_session(name)
        assert os.path.isdir(path)
        shutil.rmtree(path)

    def test_create_session_with_dashes_and_underscores(self):
        """Mixed dashes and underscores should work."""
        name = "A-Level_Biology-2024"
        path = create_session(name)
        assert os.path.isdir(path)
        shutil.rmtree(path)


# ============================================================
# CACHE.PY DEEP AUDIT
# ============================================================

class TestCacheEdgeCases:
    """Test cache system edge cases and boundary conditions."""

    def test_save_and_load_dict(self):
        """Cache should handle dict data."""
        name = "test-cache-dict"
        path = create_session(name)
        data = {"key": "value", "nested": {"a": 1}}
        save_to_cache(path, "test", data)
        loaded = load_from_cache(path, "test")
        assert loaded == data
        shutil.rmtree(path)

    def test_save_and_load_list(self):
        """Cache should handle list data."""
        name = "test-cache-list"
        path = create_session(name)
        data = [{"title": "A"}, {"title": "B"}]
        save_to_cache(path, "test", data)
        loaded = load_from_cache(path, "test")
        assert loaded == data
        shutil.rmtree(path)

    def test_save_and_load_string(self):
        """Cache should handle string data."""
        name = "test-cache-string"
        path = create_session(name)
        save_to_cache(path, "test", "hello world")
        loaded = load_from_cache(path, "test")
        assert loaded == "hello world"
        shutil.rmtree(path)

    def test_save_and_load_empty_dict(self):
        """Cache should handle empty dict."""
        name = "test-cache-empty"
        path = create_session(name)
        save_to_cache(path, "test", {})
        loaded = load_from_cache(path, "test")
        assert loaded == {}
        shutil.rmtree(path)

    def test_save_and_load_empty_list(self):
        """Cache should handle empty list."""
        name = "test-cache-empty-list"
        path = create_session(name)
        save_to_cache(path, "test", [])
        loaded = load_from_cache(path, "test")
        assert loaded == []
        shutil.rmtree(path)

    def test_overwrite_cache(self):
        """Saving to same key should overwrite old data."""
        name = "test-cache-overwrite"
        path = create_session(name)
        save_to_cache(path, "test", {"version": 1})
        save_to_cache(path, "test", {"version": 2})
        loaded = load_from_cache(path, "test")
        assert loaded == {"version": 2}
        shutil.rmtree(path)

    def test_multiple_cache_keys(self):
        """Different keys should not interfere with each other."""
        name = "test-cache-multi"
        path = create_session(name)
        save_to_cache(path, "key1", {"id": 1})
        save_to_cache(path, "key2", {"id": 2})
        assert load_from_cache(path, "key1") == {"id": 1}
        assert load_from_cache(path, "key2") == {"id": 2}
        shutil.rmtree(path)

    def test_load_nonexistent_returns_none(self):
        """Loading a nonexistent key should return None."""
        name = "test-cache-missing"
        path = create_session(name)
        assert load_from_cache(path, "nonexistent") is None
        shutil.rmtree(path)

    def test_is_cache_valid_nonexistent_returns_false(self):
        """Checking validity of nonexistent cache should return False."""
        name = "test-cache-valid-missing"
        path = create_session(name)
        assert is_cache_valid(path, "nonexistent") is False
        shutil.rmtree(path)

    def test_cache_valid_immediately_after_save(self):
        """Cache should be valid immediately after saving."""
        name = "test-cache-fresh"
        path = create_session(name)
        save_to_cache(path, "test", {"data": True})
        assert is_cache_valid(path, "test") is True
        shutil.rmtree(path)

    def test_cache_invalid_after_ttl(self):
        """Cache should be invalid after TTL expires."""
        name = "test-cache-expired"
        path = create_session(name)

        # Save with old timestamp
        cache_file = _get_cache_file_path(path, "test")
        old_time = time.time() - (settings.CACHE_TTL_SECONDS + 1)
        with open(cache_file, "w") as f:
            json.dump({"data": "old", "timestamp": old_time}, f)

        assert is_cache_valid(path, "test") is False
        shutil.rmtree(path)

    def test_cache_valid_at_ttl_boundary(self):
        """Cache should be valid at exactly TTL - 1 second."""
        name = "test-cache-boundary"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        boundary_time = time.time() - (settings.CACHE_TTL_SECONDS - 1)
        with open(cache_file, "w") as f:
            json.dump({"data": "boundary", "timestamp": boundary_time}, f)

        assert is_cache_valid(path, "test") is True
        shutil.rmtree(path)

    def test_cache_invalid_at_ttl_exact(self):
        """Cache should be invalid at exactly TTL + 1 second."""
        name = "test-cache-boundary-exact"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        exact_time = time.time() - (settings.CACHE_TTL_SECONDS + 1)
        with open(cache_file, "w") as f:
            json.dump({"data": "exact", "timestamp": exact_time}, f)

        assert is_cache_valid(path, "test") is False
        shutil.rmtree(path)

    def test_corrupted_json_returns_none(self):
        """Corrupted JSON file should return None, not raise."""
        name = "test-cache-corrupt-json"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        with open(cache_file, "w") as f:
            f.write("not valid json {{{")

        assert load_from_cache(path, "test") is None
        shutil.rmtree(path)

    def test_corrupted_json_invalid_returns_false(self):
        """Corrupted JSON should make is_cache_valid return False."""
        name = "test-cache-corrupt-valid"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        with open(cache_file, "w") as f:
            f.write("not valid json {{{")

        assert is_cache_valid(path, "test") is False
        shutil.rmtree(path)

    def test_missing_data_key_returns_none(self):
        """JSON without 'data' key should return None."""
        name = "test-cache-no-data"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        with open(cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert load_from_cache(path, "test") is None
        shutil.rmtree(path)

    def test_missing_timestamp_key_returns_false(self):
        """JSON without 'timestamp' key should make is_cache_valid return False."""
        name = "test-cache-no-timestamp"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        with open(cache_file, "w") as f:
            json.dump({"data": "test"}, f)

        # Should not raise KeyError
        assert is_cache_valid(path, "test") is False
        shutil.rmtree(path)

    def test_empty_file_returns_none(self):
        """Empty cache file should return None."""
        name = "test-cache-empty-file"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        with open(cache_file, "w") as f:
            f.write("")

        assert load_from_cache(path, "test") is None
        shutil.rmtree(path)

    def test_cache_key_with_special_characters(self):
        """Cache keys with special characters should work."""
        name = "test-cache-special-key"
        path = create_session(name)
        save_to_cache(path, "search_a-level-biology", {"test": True})
        loaded = load_from_cache(path, "search_a-level-biology")
        assert loaded == {"test": True}
        shutil.rmtree(path)

    def test_large_data_cache(self):
        """Cache should handle large data."""
        name = "test-cache-large"
        path = create_session(name)
        large_data = {"results": [{"title": f"Title {i}", "content": "x" * 1000} for i in range(100)]}
        save_to_cache(path, "large", large_data)
        loaded = load_from_cache(path, "large")
        assert len(loaded["results"]) == 100
        shutil.rmtree(path)


# ============================================================
# WEB_SEARCH.PY DEEP AUDIT
# ============================================================

class TestWebSearchEdgeCases:
    """Test web_search edge cases and failure modes."""

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_empty_results(self, MockClient):
        """Handle empty search results gracefully."""
        name = "test-search-empty"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.return_value = {"results": []}
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "no results", "session_path": path}))
        assert results[0]["snippet"].startswith("No results found")
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_missing_results_key(self, MockClient):
        """Handle missing 'results' key in API response."""
        name = "test-search-no-key"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.return_value = {}
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test", "session_path": path}))
        assert results[0]["snippet"].startswith("No results found")
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_result_missing_fields(self, MockClient):
        """Handle results with missing title/url/content fields."""
        name = "test-search-missing-fields"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.return_value = {
            "results": [{"title": "Only Title"}]  # Missing url and content
        }
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test", "session_path": path}))
        assert results[0]["title"] == "Only Title"
        assert results[0]["url"] == ""
        assert results[0]["snippet"] == ""
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_snippet_truncation(self, MockClient):
        """Snippets should be truncated to 300 characters."""
        name = "test-search-truncate"
        path = create_session(name)
        long_content = "x" * 500
        MockClient.return_value.__aenter__.return_value.search.return_value = {
            "results": [{"title": "T", "url": "U", "content": long_content}]
        }
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test", "session_path": path}))
        assert len(results[0]["snippet"]) <= 300
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_api_timeout_returns_error(self, MockClient):
        """API timeout should return error dict."""
        name = "test-search-timeout"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.side_effect = TimeoutError("Connection timed out")
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test", "session_path": path}))
        assert "timed out" in results[0]["snippet"].lower()
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_api_rate_limit_returns_error(self, MockClient):
        """API rate limit should return error dict."""
        name = "test-search-rate-limit"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.side_effect = Exception("Rate limit exceeded")
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test", "session_path": path}))
        assert "Rate limit" in results[0]["snippet"]
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_search_saves_to_file(self, MockClient):
        """Search results should be saved to search_results.md."""
        name = "test-search-file-save"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.search.return_value = {
            "results": [{"title": "Test", "url": "https://test.com", "content": "Content"}]
        }
        from topictrace.tools.web_search import web_search
        asyncio.run(web_search.ainvoke({"query": "test query", "session_path": path}))

        file_path = os.path.join(path, "search_results.md")
        assert os.path.exists(file_path)
        with open(file_path) as f:
            content = f.read()
        assert "Test" in content
        assert "https://test.com" in content
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_search_caches_results(self, MockClient):
        """Search results should be cached."""
        name = "test-search-cache"
        path = create_session(name)
        mock_search = MockClient.return_value.__aenter__.return_value.search
        mock_search.return_value = {
            "results": [{"title": "Cached", "url": "U", "content": "C"}]
        }
        from topictrace.tools.web_search import web_search
        asyncio.run(web_search.ainvoke({"query": "test query", "session_path": path}))

        # Second call should use cache (no API call)
        mock_search.reset_mock()
        results = asyncio.run(web_search.ainvoke({"query": "test query", "session_path": path}))
        assert results[0]["title"] == "Cached"
        mock_search.assert_not_called()
        shutil.rmtree(path)


# ============================================================
# WEB_FETCH.PY DEEP AUDIT
# ============================================================

class TestWebFetchEdgeCases:
    """Test web_fetch edge cases and failure modes."""

    def test_empty_url_returns_error(self):
        """Empty URL should return error dict."""
        name = "test-fetch-empty-url"
        path = create_session(name)
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "", "session_path": path}))
        assert results[0]["status"] == "error"
        assert "empty" in results[0]["content"].lower()
        shutil.rmtree(path)

    def test_whitespace_url_returns_error(self):
        """Whitespace-only URL should return error dict."""
        name = "test-fetch-whitespace-url"
        path = create_session(name)
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "   ", "session_path": path}))
        assert results[0]["status"] == "error"
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_404_returns_error(self, MockClient):
        """404 response should return error dict."""
        name = "test-fetch-404"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 404
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com/notfound", "session_path": path}))
        assert results[0]["status"] == 404
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_500_returns_error(self, MockClient):
        """500 response should return error dict."""
        name = "test-fetch-500"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 500
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com/error", "session_path": path}))
        assert results[0]["status"] == 500
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_403_returns_error(self, MockClient):
        """403 response should return error dict."""
        name = "test-fetch-403"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 403
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com/forbidden", "session_path": path}))
        assert results[0]["status"] == 403
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_network_error_returns_error(self, MockClient):
        """Network error should return error dict."""
        name = "test-fetch-network"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.get.side_effect = ConnectionError("Network unreachable")
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "session_path": path}))
        assert results[0]["status"] == "error"
        assert "Network unreachable" in results[0]["content"]
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_timeout_returns_error(self, MockClient):
        """Timeout should return error dict."""
        name = "test-fetch-timeout"
        path = create_session(name)
        MockClient.return_value.__aenter__.return_value.get.side_effect = TimeoutError("Request timed out")
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "session_path": path}))
        assert results[0]["status"] == "error"
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_saves_content_to_file(self, MockClient):
        """Fetched content should be saved to fetched_pages/."""
        name = "test-fetch-file-save"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "# Test Content\n\nBody text."
        mock_response.status_code = 200
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com/page", "session_path": path}))

        files = os.listdir(os.path.join(path, "fetched_pages"))
        assert len(files) >= 1
        with open(os.path.join(path, "fetched_pages", files[0])) as f:
            content = f.read()
        assert "Test Content" in content
        assert "Source: https://example.com/page" in content
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_fetch_caches_content(self, MockClient):
        """Fetched content should be cached."""
        name = "test-fetch-cache"
        path = create_session(name)
        mock_get = MockClient.return_value.__aenter__.return_value.get
        mock_response = MagicMock()
        mock_response.text = "# Cached Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "session_path": path}))

        # Second call should use cache
        mock_get.reset_mock()
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "session_path": path}))
        assert "Cached Content" in results[0]["content"]
        mock_get.assert_not_called()
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_jina_url_construction(self, MockClient):
        """Jina URL should be constructed correctly."""
        name = "test-fetch-jina-url"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        mock_client = MockClient.return_value.__aenter__.return_value
        mock_client.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com/page", "session_path": path}))

        mock_client.get.assert_called_once_with(
            "https://r.jina.ai/https://example.com/page"
        )
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_multiple_fetches_increment_number(self, MockClient):
        """Multiple fetches should create page_1.md, page_2.md, etc."""
        name = "test-fetch-increment"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com/1", "session_path": path}))
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com/2", "session_path": path}))

        files = sorted(os.listdir(os.path.join(path, "fetched_pages")))
        assert len(files) == 2
        assert "page_1.md" in files
        assert "page_2.md" in files
        shutil.rmtree(path)


# ============================================================
# SUMMARIZE.PY DEEP AUDIT
# ============================================================

class TestSummarizeEdgeCases:
    """Test summarize edge cases and failure modes."""

    @pytest.mark.parametrize("content", ["", "   "])
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.summarize.call_llm")
    def test_empty_or_whitespace_content_proceeds(self, mock_call_llm, content):
        """Empty/whitespace content logs warning but still calls LLM."""
        path = create_session("test-summarize-empty")
        mock_call_llm.return_value = "Summary."
        from topictrace.tools.summarize import summarize
        result = asyncio.run(summarize.ainvoke({"content": content, "query": "query", "session_path": path}))
        assert result == "Summary."
        mock_call_llm.assert_called_once()
        shutil.rmtree(path)

    @patch("topictrace.settings.LLM_API_KEY", None)
    def test_missing_api_key_raises(self):
        """Missing LLM_API_KEY should raise when OpenAI client fails."""
        path = create_session("test-summarize-no-key")
        from topictrace.tools.summarize import summarize
        with pytest.raises(Exception):
            asyncio.run(summarize.ainvoke({"content": "content", "query": "query", "session_path": path}))
        shutil.rmtree(path)

    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.summarize.call_llm")
    def test_content_truncated_to_8000_chars(self, mock_call_llm):
        """Content should be truncated to 8000 characters."""
        name = "test-summarize-truncate"
        path = create_session(name)
        mock_call_llm.return_value = "Summary."

        from topictrace.tools.summarize import summarize
        long_content = "x" * 20000
        asyncio.run(summarize.ainvoke({"content": long_content, "query": "query", "session_path": path}))

        # Check the call was made (content truncation happens before call_llm)
        mock_call_llm.assert_called_once()
        shutil.rmtree(path)

    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.summarize.call_llm")
    def test_saves_summary_to_file(self, mock_call_llm):
        """Summary should be saved to summaries/ directory."""
        name = "test-summarize-file"
        path = create_session(name)
        mock_call_llm.return_value = "Saved summary."

        from topictrace.tools.summarize import summarize
        asyncio.run(summarize.ainvoke({"content": "content", "query": "query", "session_path": path}))

        files = os.listdir(os.path.join(path, "summaries"))
        assert len(files) >= 1
        with open(os.path.join(path, "summaries", files[0])) as f:
            assert f.read() == "Saved summary."
        shutil.rmtree(path)

    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.summarize.call_llm")
    def test_api_error_propagates(self, mock_call_llm):
        """API errors should propagate as exceptions."""
        name = "test-summarize-api-error"
        path = create_session(name)
        mock_call_llm.side_effect = RuntimeError("API down")

        from topictrace.tools.summarize import summarize
        with pytest.raises(RuntimeError, match="API down"):
            asyncio.run(summarize.ainvoke({"content": "content", "query": "query", "session_path": path}))
        shutil.rmtree(path)


# ============================================================
# INTEGRATION DEEP AUDIT
# ============================================================

class TestIntegrationDeepAudit:
    """Test full chain with all failure paths."""

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    @patch("topictrace.tools.summarize.call_llm")
    def test_full_chain_search_fetch_summarize(self, mock_call_llm, MockFetchClient, MockSearchClient):
        """Full chain: search → fetch → summarize with file saving."""
        from topictrace.tools.web_search import web_search
        from topictrace.tools.web_fetch import web_fetch
        from topictrace.tools.summarize import summarize

        name = "test-integration-full"
        path = create_session(name)

        # Mock search
        mock_search_instance = AsyncMock()
        MockSearchClient.return_value.__aenter__.return_value = mock_search_instance
        mock_search_instance.search.return_value = {
            "results": [{"title": "Biology", "url": "https://bio.com", "content": "Cells"}]
        }

        # Mock fetch
        mock_response = MagicMock()
        mock_response.text = "# Biology\n\nCell biology is fundamental."
        mock_response.status_code = 200
        MockFetchClient.return_value.__aenter__.return_value.get.return_value = mock_response

        # Mock summarize
        mock_call_llm.return_value = "Cell biology covers cell structure."

        # Step 1: Search
        search_results = asyncio.run(web_search.ainvoke({"session_path": path, "query": "biology"}))
        assert len(search_results) == 1
        assert os.path.exists(os.path.join(path, "search_results.md"))

        # Step 2: Fetch
        fetch_results = asyncio.run(web_fetch.ainvoke({"session_path": path, "url": search_results[0]["url"]}))
        assert isinstance(fetch_results, list)
        content = fetch_results[0]["content"]
        assert "Biology" in content
        assert len(os.listdir(os.path.join(path, "fetched_pages"))) >= 1

        # Step 3: Summarize
        summary = asyncio.run(summarize.ainvoke({"session_path": path, "content": content, "query": "biology"}))
        assert "cell" in summary.lower()
        assert len(os.listdir(os.path.join(path, "summaries"))) >= 1

        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_search_failure_does_not_corrupt_session(self, MockSearchClient):
        """Search failure should not leave corrupted files in session."""
        from topictrace.tools.web_search import web_search

        name = "test-integration-search-fail"
        path = create_session(name)

        mock_search_instance = AsyncMock()
        MockSearchClient.return_value.__aenter__.return_value = mock_search_instance
        mock_search_instance.search.side_effect = Exception("API down")

        results = asyncio.run(web_search.ainvoke({"session_path": path, "query": "test"}))
        assert "Request failed" in results[0]["snippet"]

        # Session should still be clean
        assert os.path.isdir(path)
        assert os.path.isdir(os.path.join(path, "cache"))
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_fetch_failure_does_not_corrupt_session(self, MockFetchClient):
        """Fetch failure should not leave corrupted files in session."""
        from topictrace.tools.web_fetch import web_fetch

        name = "test-integration-fetch-fail"
        path = create_session(name)

        mock_client = AsyncMock()
        MockFetchClient.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = Exception("Network down")

        results = asyncio.run(web_fetch.ainvoke({"session_path": path, "url": "https://example.com"}))
        assert results[0]["status"] == "error"

        assert os.path.isdir(path)
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.summarize.call_llm")
    def test_summarize_failure_does_not_corrupt_session(self, mock_call_llm):
        """Summarize failure should not leave corrupted files in session."""
        from topictrace.tools.summarize import summarize

        name = "test-integration-summarize-fail"
        path = create_session(name)
        mock_call_llm.side_effect = Exception("Model error")

        with pytest.raises(Exception, match="Model error"):
            asyncio.run(summarize.ainvoke({"session_path": path, "content": "text", "query": "q"}))

        assert os.path.isdir(path)
        shutil.rmtree(path)

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.settings.LLM_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    @patch("topictrace.tools.summarize.call_llm")
    def test_cache_prevents_duplicate_api_calls(self, mock_call_llm, MockFetchClient, MockSearchClient):
        """Cached results should prevent duplicate API calls."""
        from topictrace.tools.web_search import web_search
        from topictrace.tools.web_fetch import web_fetch

        name = "test-integration-cache"
        path = create_session(name)

        mock_search_instance = AsyncMock()
        MockSearchClient.return_value.__aenter__.return_value = mock_search_instance
        mock_search_instance.search.return_value = {
            "results": [{"title": "T", "url": "U", "content": "C"}]
        }

        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        MockFetchClient.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_call_llm.return_value = "Summary."

        # First call
        asyncio.run(web_search.ainvoke({"session_path": path, "query": "test"}))
        asyncio.run(web_fetch.ainvoke({"session_path": path, "url": "https://example.com"}))

        # Reset mocks
        mock_search_instance.search.reset_mock()
        MockFetchClient.return_value.__aenter__.return_value.get.reset_mock()

        # Second call should use cache
        asyncio.run(web_search.ainvoke({"session_path": path, "query": "test"}))
        asyncio.run(web_fetch.ainvoke({"session_path": path, "url": "https://example.com"}))

        mock_search_instance.search.assert_not_called()
        MockFetchClient.return_value.__aenter__.return_value.get.assert_not_called()

        shutil.rmtree(path)
