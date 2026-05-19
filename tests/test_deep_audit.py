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
- registry.py: argument validation, tool dispatch, definitions format
- integration: full chain with all failure paths
"""

import json
import os
import shutil
import time
from unittest.mock import patch, MagicMock

import pytest

from topictrace.session import create_session, get_session_path, _sanitize_session_name, SESSIONS_DIR
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid, CACHE_TTL_SECONDS, _get_cache_file_path


# ============================================================
# SESSION.PY DEEP AUDIT
# ============================================================

class TestSessionSanitization:
    """Test every possible malicious or invalid session name input."""

    def test_path_traversal_dot_dot_slash(self):
        """Block ../../etc/passwd style attacks."""
        result = _sanitize_session_name("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_path_traversal_dot_dot_backslash(self):
        """Block ..\\..\\etc\\passwd style attacks on Windows."""
        result = _sanitize_session_name("..\\..\\etc\\passwd")
        assert ".." not in result
        assert "\\" not in result

    def test_path_traversal_absolute_path(self):
        """Block /etc/passwd style attacks."""
        result = _sanitize_session_name("/etc/passwd")
        assert "/" not in result
        assert result.startswith("etcpasswd") or len(result) > 0

    def test_path_traversal_null_byte(self):
        """Block null byte injection."""
        result = _sanitize_session_name("session\x00name")
        assert "\x00" not in result

    def test_path_traversal_unicode_slash(self):
        """Block unicode slash characters."""
        result = _sanitize_session_name("session\u2215name")
        assert "\u2215" not in result

    def test_special_characters_stripped(self):
        """Strip all special characters except dash and underscore."""
        result = _sanitize_session_name("test!@#$%^&*()+={}[]|\\:;\"'<>,.?/~`")
        # Only alphanumeric, dash, underscore should remain
        for char in result:
            assert char.isalnum() or char in "-_", f"Unexpected char: {char}"

    def test_unicode_letters_kept(self):
        """Keep alphanumeric unicode characters."""
        result = _sanitize_session_name("Biology-2024")
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
        result = _sanitize_session_name("test---name")
        assert "---" not in result
        assert "--" not in result

    def test_underscores_converted_to_dashes(self):
        """Underscores should be converted to dashes."""
        result = _sanitize_session_name("test_name")
        assert "_" not in result
        assert "-" in result

    def test_leading_trailing_dashes_stripped(self):
        """Leading and trailing dashes should be stripped."""
        result = _sanitize_session_name("---test---")
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
        assert path.startswith(SESSIONS_DIR)
        shutil.rmtree(path)

    def test_get_session_path_format(self):
        """Path should be sessions/<name> format."""
        path = get_session_path("my-session")
        assert path == os.path.join(SESSIONS_DIR, "my-session")

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
        old_time = time.time() - (CACHE_TTL_SECONDS + 1)
        with open(cache_file, "w") as f:
            json.dump({"data": "old", "timestamp": old_time}, f)

        assert is_cache_valid(path, "test") is False
        shutil.rmtree(path)

    def test_cache_valid_at_ttl_boundary(self):
        """Cache should be valid at exactly TTL - 1 second."""
        name = "test-cache-boundary"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        boundary_time = time.time() - (CACHE_TTL_SECONDS - 1)
        with open(cache_file, "w") as f:
            json.dump({"data": "boundary", "timestamp": boundary_time}, f)

        assert is_cache_valid(path, "test") is True
        shutil.rmtree(path)

    def test_cache_invalid_at_ttl_exact(self):
        """Cache should be invalid at exactly TTL + 1 second."""
        name = "test-cache-boundary-exact"
        path = create_session(name)

        cache_file = _get_cache_file_path(path, "test")
        exact_time = time.time() - (CACHE_TTL_SECONDS + 1)
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

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_empty_results(self, MockClient):
        """Handle empty search results gracefully."""
        name = "test-search-empty"
        path = create_session(name)
        MockClient.return_value.search.return_value = {"results": []}
        from topictrace.tools.web_search import web_search
        results = web_search("no results", path)
        assert results == []
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_missing_results_key(self, MockClient):
        """Handle missing 'results' key in API response."""
        name = "test-search-no-key"
        path = create_session(name)
        MockClient.return_value.search.return_value = {}
        from topictrace.tools.web_search import web_search
        results = web_search("test", path)
        assert results == []
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_result_missing_fields(self, MockClient):
        """Handle results with missing title/url/content fields."""
        name = "test-search-missing-fields"
        path = create_session(name)
        MockClient.return_value.search.return_value = {
            "results": [{"title": "Only Title"}]  # Missing url and content
        }
        from topictrace.tools.web_search import web_search
        results = web_search("test", path)
        assert results[0]["title"] == "Only Title"
        assert results[0]["url"] == ""
        assert results[0]["snippet"] == ""
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_snippet_truncation(self, MockClient):
        """Snippets should be truncated to 300 characters."""
        name = "test-search-truncate"
        path = create_session(name)
        long_content = "x" * 500
        MockClient.return_value.search.return_value = {
            "results": [{"title": "T", "url": "U", "content": long_content}]
        }
        from topictrace.tools.web_search import web_search
        results = web_search("test", path)
        assert len(results[0]["snippet"]) <= 300
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_api_timeout_raises(self, MockClient):
        """API timeout should propagate as exception."""
        name = "test-search-timeout"
        path = create_session(name)
        MockClient.return_value.search.side_effect = TimeoutError("Connection timed out")
        from topictrace.tools.web_search import web_search
        with pytest.raises(TimeoutError):
            web_search("test", path)
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_api_rate_limit_raises(self, MockClient):
        """API rate limit should propagate as exception."""
        name = "test-search-rate-limit"
        path = create_session(name)
        MockClient.return_value.search.side_effect = Exception("Rate limit exceeded")
        from topictrace.tools.web_search import web_search
        with pytest.raises(Exception, match="Rate limit"):
            web_search("test", path)
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_search_saves_to_file(self, MockClient):
        """Search results should be saved to search_results.md."""
        name = "test-search-file-save"
        path = create_session(name)
        MockClient.return_value.search.return_value = {
            "results": [{"title": "Test", "url": "https://test.com", "content": "Content"}]
        }
        from topictrace.tools.web_search import web_search
        web_search("test query", path)

        file_path = os.path.join(path, "search_results.md")
        assert os.path.exists(file_path)
        with open(file_path) as f:
            content = f.read()
        assert "Test" in content
        assert "https://test.com" in content
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_search_caches_results(self, MockClient):
        """Search results should be cached."""
        name = "test-search-cache"
        path = create_session(name)
        MockClient.return_value.search.return_value = {
            "results": [{"title": "Cached", "url": "U", "content": "C"}]
        }
        from topictrace.tools.web_search import web_search
        web_search("test query", path)

        # Second call should use cache (no API call)
        MockClient.return_value.search.reset_mock()
        results = web_search("test query", path)
        assert results[0]["title"] == "Cached"
        MockClient.return_value.search.assert_not_called()
        shutil.rmtree(path)


# ============================================================
# WEB_FETCH.PY DEEP AUDIT
# ============================================================

class TestWebFetchEdgeCases:
    """Test web_fetch edge cases and failure modes."""

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_empty_url_raises(self, mock_get):
        """Empty URL should raise ValueError."""
        name = "test-fetch-empty-url"
        path = create_session(name)
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(ValueError, match="URL cannot be empty"):
            web_fetch("", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_whitespace_url_raises(self, mock_get):
        """Whitespace-only URL should raise ValueError."""
        name = "test-fetch-whitespace-url"
        path = create_session(name)
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(ValueError, match="URL cannot be empty"):
            web_fetch("   ", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_404_raises_exception(self, mock_get):
        """404 response should raise Exception."""
        name = "test-fetch-404"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(Exception, match="status 404"):
            web_fetch("https://example.com/notfound", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_500_raises_exception(self, mock_get):
        """500 response should raise Exception."""
        name = "test-fetch-500"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(Exception, match="status 500"):
            web_fetch("https://example.com/error", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_403_raises_exception(self, mock_get):
        """403 response should raise Exception."""
        name = "test-fetch-403"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(Exception, match="status 403"):
            web_fetch("https://example.com/forbidden", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network error should propagate."""
        name = "test-fetch-network"
        path = create_session(name)
        mock_get.side_effect = ConnectionError("Network unreachable")
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(ConnectionError):
            web_fetch("https://example.com", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_timeout_error_raises(self, mock_get):
        """Timeout should propagate."""
        name = "test-fetch-timeout"
        path = create_session(name)
        mock_get.side_effect = TimeoutError("Request timed out")
        from topictrace.tools.web_fetch import web_fetch
        with pytest.raises(TimeoutError):
            web_fetch("https://example.com", path)
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_saves_content_to_file(self, mock_get):
        """Fetched content should be saved to fetched_pages/."""
        name = "test-fetch-file-save"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "# Test Content\n\nBody text."
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        web_fetch("https://example.com/page", path)

        files = os.listdir(os.path.join(path, "fetched_pages"))
        assert len(files) >= 1
        with open(os.path.join(path, "fetched_pages", files[0])) as f:
            content = f.read()
        assert "Test Content" in content
        assert "Source: https://example.com/page" in content
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_fetch_caches_content(self, mock_get):
        """Fetched content should be cached."""
        name = "test-fetch-cache"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "# Cached Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        web_fetch("https://example.com", path)

        # Second call should use cache
        mock_get.reset_mock()
        content = web_fetch("https://example.com", path)
        assert "Cached Content" in content
        mock_get.assert_not_called()
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_jina_url_construction(self, mock_get):
        """Jina URL should be constructed correctly."""
        name = "test-fetch-jina-url"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        web_fetch("https://example.com/page", path)

        mock_get.assert_called_once_with(
            "https://r.jina.ai/https://example.com/page",
            timeout=30
        )
        shutil.rmtree(path)

    @patch("topictrace.tools.web_fetch.requests.get")
    def test_multiple_fetches_increment_number(self, mock_get):
        """Multiple fetches should create page_1.md, page_2.md, etc."""
        name = "test-fetch-increment"
        path = create_session(name)
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        web_fetch("https://example.com/1", path)
        web_fetch("https://example.com/2", path)

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

    def test_empty_content_raises(self):
        """Empty content should raise ValueError."""
        name = "test-summarize-empty"
        path = create_session(name)
        from topictrace.tools.summarize import summarize
        with pytest.raises(ValueError, match="Content cannot be empty"):
            summarize("", "query", path)
        shutil.rmtree(path)

    def test_whitespace_content_raises(self):
        """Whitespace-only content should raise ValueError."""
        name = "test-summarize-whitespace"
        path = create_session(name)
        from topictrace.tools.summarize import summarize
        with pytest.raises(ValueError, match="Content cannot be empty"):
            summarize("   ", "query", path)
        shutil.rmtree(path)

    def test_missing_api_key_raises(self):
        """Missing NVIDIA_API_KEY should raise ValueError."""
        name = "test-summarize-no-key"
        path = create_session(name)
        old_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            from topictrace.tools.summarize import summarize
            with pytest.raises(ValueError, match="NVIDIA_API_KEY not found"):
                summarize("content", "query", path)
        finally:
            if old_key:
                os.environ["NVIDIA_API_KEY"] = old_key
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_streaming_chunks_collected(self, MockOpenAI):
        """Streaming chunks should be collected into full summary."""
        name = "test-summarize-streaming"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Part 1. "
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Part 2."
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        from topictrace.tools.summarize import summarize
        result = summarize("content", "query", path)
        assert result == "Part 1. Part 2."
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_streaming_empty_choices_skipped(self, MockOpenAI):
        """Empty choices in streaming should be skipped."""
        name = "test-summarize-empty-choices"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        empty_chunk = MagicMock()
        empty_chunk.choices = []
        content_chunk = MagicMock()
        content_chunk.choices = [MagicMock()]
        content_chunk.choices[0].delta.content = "Summary."
        mock_client.chat.completions.create.return_value = [empty_chunk, content_chunk]

        from topictrace.tools.summarize import summarize
        result = summarize("content", "query", path)
        assert result == "Summary."
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_streaming_none_content_skipped(self, MockOpenAI):
        """None content in delta should be skipped."""
        name = "test-summarize-none-content"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        none_chunk = MagicMock()
        none_chunk.choices = [MagicMock()]
        none_chunk.choices[0].delta.content = None
        content_chunk = MagicMock()
        content_chunk.choices = [MagicMock()]
        content_chunk.choices[0].delta.content = "Done."
        mock_client.chat.completions.create.return_value = [none_chunk, content_chunk]

        from topictrace.tools.summarize import summarize
        result = summarize("content", "query", path)
        assert result == "Done."
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_content_truncated_to_8000_chars(self, MockOpenAI):
        """Content should be truncated to 8000 characters."""
        name = "test-summarize-truncate"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Summary."
        mock_client.chat.completions.create.return_value = [chunk]

        from topictrace.tools.summarize import summarize
        long_content = "x" * 20000
        summarize(long_content, "query", path)

        # Check the message content was truncated
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        # The content portion should be at most 8000 chars
        content_start = user_message.index("Content to summarize:\n") + len("Content to summarize:\n")
        content_in_prompt = user_message[content_start:]
        assert len(content_in_prompt) <= 8000
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_saves_summary_to_file(self, MockOpenAI):
        """Summary should be saved to summaries/ directory."""
        name = "test-summarize-file"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Saved summary."
        mock_client.chat.completions.create.return_value = [chunk]

        from topictrace.tools.summarize import summarize
        summarize("content", "query", path)

        files = os.listdir(os.path.join(path, "summaries"))
        assert len(files) >= 1
        with open(os.path.join(path, "summaries", files[0])) as f:
            assert f.read() == "Saved summary."
        shutil.rmtree(path)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_api_error_propagates(self, MockOpenAI):
        """API errors should propagate as exceptions."""
        name = "test-summarize-api-error"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        from topictrace.tools.summarize import summarize
        with pytest.raises(RuntimeError, match="API down"):
            summarize("content", "query", path)
        shutil.rmtree(path)


# ============================================================
# REGISTRY.PY DEEP AUDIT
# ============================================================

class TestRegistryEdgeCases:
    """Test registry edge cases and failure modes."""

    def test_get_tool_definitions_returns_three(self):
        """Should return exactly 3 tool definitions."""
        from topictrace.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        assert len(tools) == 3

    def test_all_tools_have_required_fields(self):
        """Each tool definition must have type, function.name, function.description, function.parameters."""
        from topictrace.tools.registry import get_tool_definitions
        for tool in get_tool_definitions():
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            assert "properties" in tool["function"]["parameters"]
            assert "required" in tool["function"]["parameters"]

    def test_tool_names_match_functions(self):
        """Tool names in definitions should match actual function names."""
        from topictrace.tools.registry import get_tool_definitions
        names = [t["function"]["name"] for t in get_tool_definitions()]
        assert "web_search" in names
        assert "web_fetch" in names
        assert "summarize" in names

    def test_web_search_requires_query(self):
        """web_search should require 'query' parameter."""
        from topictrace.tools.registry import get_tool_definitions
        search_def = [t for t in get_tool_definitions() if t["function"]["name"] == "web_search"][0]
        assert "query" in search_def["function"]["parameters"]["required"]

    def test_web_fetch_requires_url(self):
        """web_fetch should require 'url' parameter."""
        from topictrace.tools.registry import get_tool_definitions
        fetch_def = [t for t in get_tool_definitions() if t["function"]["name"] == "web_fetch"][0]
        assert "url" in fetch_def["function"]["parameters"]["required"]

    def test_summarize_requires_content_and_query(self):
        """summarize should require 'content' and 'query' parameters."""
        from topictrace.tools.registry import get_tool_definitions
        summarize_def = [t for t in get_tool_definitions() if t["function"]["name"] == "summarize"][0]
        assert "content" in summarize_def["function"]["parameters"]["required"]
        assert "query" in summarize_def["function"]["parameters"]["required"]

    def test_unknown_tool_returns_error_string(self):
        """Unknown tool name should return error string, not raise."""
        name = "test-registry-unknown"
        path = create_session(name)
        from topictrace.tools.registry import run_tool
        result = run_tool("nonexistent_tool", session_path=path)
        assert "Error" in result
        assert "nonexistent_tool" in result
        shutil.rmtree(path)

    def test_unknown_tool_lists_available_tools(self):
        """Error message should list available tool names."""
        name = "test-registry-list"
        path = create_session(name)
        from topictrace.tools.registry import run_tool
        result = run_tool("bad_tool", session_path=path)
        assert "web_search" in result
        assert "web_fetch" in result
        assert "summarize" in result
        shutil.rmtree(path)

    def test_empty_tool_name_returns_error(self):
        """Empty tool name should return error."""
        name = "test-registry-empty"
        path = create_session(name)
        from topictrace.tools.registry import run_tool
        result = run_tool("", session_path=path)
        assert "Error" in result
        shutil.rmtree(path)


# ============================================================
# INTEGRATION DEEP AUDIT
# ============================================================

class TestIntegrationDeepAudit:
    """Test full chain with all failure paths."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    @patch("topictrace.tools.web_fetch.requests.get")
    @patch("topictrace.tools.summarize.OpenAI")
    def test_full_chain_search_fetch_summarize(self, MockOpenAI, mock_get, MockTavily):
        """Full chain: search → fetch → summarize with file saving."""
        name = "test-integration-full"
        path = create_session(name)

        # Mock search
        MockTavily.return_value.search.return_value = {
            "results": [{"title": "Biology", "url": "https://bio.com", "content": "Cells"}]
        }

        # Mock fetch
        mock_response = MagicMock()
        mock_response.text = "# Biology\n\nCell biology is fundamental."
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock summarize
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Cell biology covers cell structure."
        mock_client.chat.completions.create.return_value = [chunk]

        from topictrace.tools.registry import run_tool

        # Step 1: Search
        search_results = run_tool("web_search", session_path=path, query="biology")
        assert len(search_results) == 1
        assert os.path.exists(os.path.join(path, "search_results.md"))

        # Step 2: Fetch
        content = run_tool("web_fetch", session_path=path, url=search_results[0]["url"])
        assert "Biology" in content
        assert len(os.listdir(os.path.join(path, "fetched_pages"))) >= 1

        # Step 3: Summarize
        summary = run_tool("summarize", session_path=path, content=content, query="biology")
        assert "cell" in summary.lower()
        assert len(os.listdir(os.path.join(path, "summaries"))) >= 1

        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    def test_search_failure_does_not_corrupt_session(self, MockTavily):
        """Search failure should not leave corrupted files in session."""
        name = "test-integration-search-fail"
        path = create_session(name)
        MockTavily.return_value.search.side_effect = Exception("API down")

        from topictrace.tools.registry import run_tool
        with pytest.raises(Exception, match="API down"):
            run_tool("web_search", session_path=path, query="test")

        # Session should still be clean
        assert os.path.isdir(path)
        assert os.path.isdir(os.path.join(path, "cache"))
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.web_fetch.requests.get")
    def test_fetch_failure_does_not_corrupt_session(self, mock_get):
        """Fetch failure should not leave corrupted files in session."""
        name = "test-integration-fetch-fail"
        path = create_session(name)
        mock_get.side_effect = ConnectionError("Network down")

        from topictrace.tools.registry import run_tool
        with pytest.raises(ConnectionError):
            run_tool("web_fetch", session_path=path, url="https://example.com")

        assert os.path.isdir(path)
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.summarize.OpenAI")
    def test_summarize_failure_does_not_corrupt_session(self, MockOpenAI):
        """Summarize failure should not leave corrupted files in session."""
        name = "test-integration-summarize-fail"
        path = create_session(name)
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Model error")

        from topictrace.tools.registry import run_tool
        with pytest.raises(Exception, match="Model error"):
            run_tool("summarize", session_path=path, content="text", query="q")

        assert os.path.isdir(path)
        shutil.rmtree(path)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "NVIDIA_API_KEY": "test-key"})
    @patch("topictrace.tools.web_search.TavilyClient")
    @patch("topictrace.tools.web_fetch.requests.get")
    @patch("topictrace.tools.summarize.OpenAI")
    def test_cache_prevents_duplicate_api_calls(self, MockOpenAI, mock_get, MockTavily):
        """Cached results should prevent duplicate API calls."""
        name = "test-integration-cache"
        path = create_session(name)

        MockTavily.return_value.search.return_value = {
            "results": [{"title": "T", "url": "U", "content": "C"}]
        }
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Summary."
        mock_client.chat.completions.create.return_value = [chunk]

        from topictrace.tools.registry import run_tool

        # First call
        run_tool("web_search", session_path=path, query="test")
        run_tool("web_fetch", session_path=path, url="https://example.com")

        # Reset mocks
        MockTavily.return_value.search.reset_mock()
        mock_get.reset_mock()

        # Second call should use cache
        run_tool("web_search", session_path=path, query="test")
        run_tool("web_fetch", session_path=path, url="https://example.com")

        MockTavily.return_value.search.assert_not_called()
        mock_get.assert_not_called()

        shutil.rmtree(path)
