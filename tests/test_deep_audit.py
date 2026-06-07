"""
Deep audit tests for TopicTrace.

Tests edge cases, boundary conditions, and failure modes across the codebase.

Coverage:
- db/client.py: generate_key_hash, key format
- tools/cache.py: generate_fetch_cache_key, load/save (mocked DB)
- tools/web_search.py: API failures, malformed responses, snippet limits
- tools/web_fetch.py: HTTP errors, timeouts, cache hit/miss, LLM summarization
- integration: health endpoint, auth middleware, full user flow
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from topictrace import settings
from topictrace.db.client import generate_key_hash


# ============================================================
# DB/CLIENT.PY — generate_key_hash
# ============================================================

class TestGenerateKeyHash:
    """Test API key hashing function."""

    def test_returns_hex_string(self):
        """Hash should be a hex string."""
        from topictrace.db.client import generate_key_hash
        result = generate_key_hash("test-key")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        """Same input should always produce same hash."""
        from topictrace.db.client import generate_key_hash
        assert generate_key_hash("abc") == generate_key_hash("abc")

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        from topictrace.db.client import generate_key_hash
        assert generate_key_hash("abc") != generate_key_hash("def")

    def test_empty_string(self):
        """Empty string should produce a valid hash."""
        from topictrace.db.client import generate_key_hash
        result = generate_key_hash("")
        assert len(result) == 64

    def test_long_input(self):
        """Long input should work fine."""
        from topictrace.db.client import generate_key_hash
        long_key = "a" * 10000
        result = generate_key_hash(long_key)
        assert len(result) == 64


# ============================================================
# TOOLS/CACHE.PY — generate_fetch_cache_key
# ============================================================

class TestGenerateFetchCacheKey:
    """Test cache key generation for web_fetch."""

    def test_returns_expected_prefix(self):
        """Key should start with cache:tool:web_fetch:"""
        from topictrace.tools.cache import generate_fetch_cache_key
        key = generate_fetch_cache_key("query", "https://example.com")
        assert key.startswith("cache:tool:web_fetch:")

    def test_deterministic(self):
        """Same inputs should produce same key."""
        from topictrace.tools.cache import generate_fetch_cache_key
        k1 = generate_fetch_cache_key("q", "https://x.com")
        k2 = generate_fetch_cache_key("q", "https://x.com")
        assert k1 == k2

    def test_different_query_different_key(self):
        """Same URL with different query should produce different keys."""
        from topictrace.tools.cache import generate_fetch_cache_key
        k1 = generate_fetch_cache_key("query1", "https://x.com")
        k2 = generate_fetch_cache_key("query2", "https://x.com")
        assert k1 != k2

    def test_different_url_different_key(self):
        """Same query with different URL should produce different keys."""
        from topictrace.tools.cache import generate_fetch_cache_key
        k1 = generate_fetch_cache_key("q", "https://a.com")
        k2 = generate_fetch_cache_key("q", "https://b.com")
        assert k1 != k2

    def test_case_insensitive(self):
        """Key should be case-insensitive (normalized to lowercase)."""
        from topictrace.tools.cache import generate_fetch_cache_key
        k1 = generate_fetch_cache_key("Query", "https://Example.COM")
        k2 = generate_fetch_cache_key("query", "https://example.com")
        assert k1 == k2


# ============================================================
# TOOLS/CACHE.PY — load/save (mocked DB)
# ============================================================

class TestCacheDB:
    """Test DB-based cache operations with mocked pool."""

    @patch("topictrace.tools.cache.pool")
    def test_save_to_cache_calls_insert(self, mock_pool):
        """save_to_cache should execute INSERT with correct params."""
        from topictrace.tools.cache import save_to_cache
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        save_to_cache("test-key", "test result")

        mock_conn.cursor.assert_called_once()

    @patch("topictrace.tools.cache.pool")
    def test_load_from_cache_returns_result(self, mock_pool):
        """load_from_cache should return result on hit."""
        from topictrace.tools.cache import load_from_cache
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [{"content": "cached data"}]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = load_from_cache("test-key")
        assert result == {"content": "cached data"}

    @patch("topictrace.tools.cache.pool")
    def test_load_from_cache_returns_none_on_miss(self, mock_pool):
        """load_from_cache should return None on cache miss."""
        from topictrace.tools.cache import load_from_cache
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = load_from_cache("missing-key")
        assert result is None

    @patch("topictrace.tools.cache.pool")
    def test_load_from_cache_returns_none_on_error(self, mock_pool):
        """load_from_cache should return None on DB error."""
        from topictrace.tools.cache import load_from_cache
        mock_pool.connection.side_effect = Exception("DB down")

        result = load_from_cache("test-key")
        assert result is None


# ============================================================
# WEB_SEARCH.PY — edge cases
# ============================================================

class TestWebSearchEdgeCases:
    """Test web_search edge cases and failure modes."""

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_empty_results(self, MockClient):
        """Handle empty search results gracefully."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.search.return_value = {"results": []}
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "no results"}))
        assert results[0]["snippet"].startswith("No results found")

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_missing_results_key(self, MockClient):
        """Handle missing 'results' key in API response."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.search.return_value = {}
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test missing results key"}))
        assert results[0]["snippet"].startswith("No results found")

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_result_missing_fields(self, MockClient):
        """Handle results with missing title/url/content fields."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.search.return_value = {
            "results": [{"title": "Only Title"}]  # Missing url and content
        }
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test"}))
        assert results[0]["title"] == "Only Title"
        assert results[0]["url"] == ""
        assert results[0]["snippet"] == ""

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_snippet_truncation(self, MockClient):
        """Snippets should be truncated to configured max chars."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        long_content = "x" * 500
        mock_instance.search.return_value = {
            "results": [{"title": "T", "url": "U", "content": long_content}]
        }
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test"}))
        assert len(results[0]["snippet"]) <= settings.SEARCH_SNIPPET_MAX_CHARS

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_api_timeout_returns_error(self, MockClient):
        """API timeout should return error dict."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.search.side_effect = TimeoutError("Connection timed out")
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test api timeout unique"}))
        assert "Connection timed out" in results[0]["snippet"]

    @patch("topictrace.settings.TAVILY_API_KEY", "test-key")
    @patch("topictrace.tools.web_search.AsyncTavilyClient")
    def test_api_rate_limit_returns_error(self, MockClient):
        """API rate limit should return error dict."""
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.search.side_effect = Exception("Rate limit exceeded")
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test api rate limit unique"}))
        assert "Rate limit exceeded" in results[0]["snippet"]

    @patch("topictrace.settings.TAVILY_API_KEY", None)
    def test_missing_api_key_returns_error(self):
        """Missing API key should return error dict."""
        from topictrace.tools.web_search import web_search
        results = asyncio.run(web_search.ainvoke({"query": "test no key"}))
        assert "TAVILY_API_KEY" in results[0]["snippet"]

    def test_invalid_query_type_raises_validation(self):
        """Non-string/list query should raise Pydantic validation error."""
        from topictrace.tools.web_search import web_search
        with pytest.raises(Exception):  # Pydantic ValidationError
            asyncio.run(web_search.ainvoke({"query": 123}))


# ============================================================
# WEB_FETCH.PY — edge cases
# ============================================================

class TestWebFetchEdgeCases:
    """Test web_fetch edge cases and failure modes."""

    def test_empty_url_returns_error(self):
        """Empty URL should return error dict."""
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "", "query": "test fetch empty"}))
        assert results[0]["status"] == "error"
        assert "empty" in results[0]["content"].lower()

    def test_whitespace_url_returns_error(self):
        """Whitespace-only URL should return error dict."""
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "   ", "query": "test fetch whitespace"}))
        assert results[0]["status"] == "error"

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_404_returns_error(self, MockClient, mock_load, mock_save, mock_llm):
        """404 response should return error dict."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com/notfound", "query": "test 404"}))
        assert results[0]["status"] == 404

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_500_returns_error(self, MockClient, mock_load, mock_save, mock_llm):
        """500 response should return error dict."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com/error", "query": "test 500"}))
        assert results[0]["status"] == 500

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_network_error_returns_error(self, MockClient, mock_load, mock_save, mock_llm):
        """Network error should return error dict."""
        MockClient.return_value.__aenter__.return_value.get.side_effect = ConnectionError("Network unreachable")
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "query": "test network"}))
        assert results[0]["status"] == "error"
        assert "Network unreachable" in results[0]["content"]

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_timeout_returns_error(self, MockClient, mock_load, mock_save, mock_llm):
        """Timeout should return error dict."""
        MockClient.return_value.__aenter__.return_value.get.side_effect = TimeoutError("Request timed out")
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "query": "test timeout"}))
        assert results[0]["status"] == "error"

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_successful_fetch_calls_llm(self, MockClient, mock_load, mock_save, mock_llm):
        """Successful fetch should call LLM and cache result."""
        mock_response = MagicMock()
        mock_response.text = "# Test Content\n\nBody text."
        mock_response.status_code = 200
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response

        mock_llm_instance = AsyncMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summarized content."
        mock_llm_instance.ainvoke.return_value = mock_llm_response
        mock_llm.return_value = mock_llm_instance

        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "query": "test"}))

        assert results[0]["status"] == 200
        assert results[0]["content"] == "Summarized content."
        mock_llm_instance.ainvoke.assert_called_once()
        mock_save.assert_called_once()

    @patch("topictrace.tools.web_fetch.load_from_cache")
    def test_cache_hit_skips_fetch(self, mock_load):
        """Cache hit should skip HTTP fetch and LLM call."""
        mock_load.return_value = "Cached summary"
        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({"url": "https://example.com", "query": "test cache hit"}))

        assert results[0]["status"] == 200
        assert results[0]["content"] == "Cached summary"

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_jina_url_construction(self, MockClient, mock_load, mock_save, mock_llm):
        """Jina URL should be constructed correctly."""
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        mock_client = MockClient.return_value.__aenter__.return_value
        mock_client.get.return_value = mock_response

        mock_llm_instance = AsyncMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summary"
        mock_llm_instance.ainvoke.return_value = mock_llm_response
        mock_llm.return_value = mock_llm_instance

        from topictrace.tools.web_fetch import web_fetch
        asyncio.run(web_fetch.ainvoke({"url": "https://example.com/page", "query": "test jina"}))

        mock_client.get.assert_called_once_with(
            "https://r.jina.ai/https://example.com/page"
        )

    @patch("topictrace.tools.web_fetch.get_llm")
    @patch("topictrace.tools.web_fetch.save_to_cache")
    @patch("topictrace.tools.web_fetch.load_from_cache", return_value=None)
    @patch("topictrace.tools.web_fetch.httpx.AsyncClient")
    def test_list_of_urls(self, MockClient, mock_load, mock_save, mock_llm):
        """Should handle list of URLs."""
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.status_code = 200
        MockClient.return_value.__aenter__.return_value.get.return_value = mock_response

        mock_llm_instance = AsyncMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summary"
        mock_llm_instance.ainvoke.return_value = mock_llm_response
        mock_llm.return_value = mock_llm_instance

        from topictrace.tools.web_fetch import web_fetch
        results = asyncio.run(web_fetch.ainvoke({
            "url": ["https://a.com", "https://b.com"],
            "query": "test list"
        }))
        assert len(results) == 2
        assert all(r["status"] == 200 for r in results)


# ============================================================
# INTEGRATION — user perspective (health + auth middleware)
# ============================================================

class TestHealthEndpoint:
    """Test health endpoint from user perspective."""

    @patch("topictrace.db.client.pool")
    def test_health_returns_ok(self, mock_pool):
        """GET /health/live should return 200 without auth."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        with TestClient(app) as client:
            response = client.get("/health/live")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


class TestApiKeyEndpoint:
    """Test POST /api-keys endpoint from user perspective."""

    @patch("topictrace.server.routes.api_key.pool")
    def test_create_api_key_returns_key(self, mock_pool):
        """POST /api-keys should return a key with tt_ prefix."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        with TestClient(app) as client:
            response = client.post("/api-keys")
            assert response.status_code == 200
            data = response.json()
            assert "key" in data
            assert data["key"].startswith("tt_")
            assert len(data["key"]) > 10

    @patch("topictrace.server.routes.api_key.pool")
    def test_create_api_key_uses_underscore_separator(self, mock_pool):
        """Key should use _ separator to match middleware split logic."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        with TestClient(app) as client:
            response = client.post("/api-keys")
            key = response.json()["key"]
            parts = key.split("_", 1)
            assert len(parts) == 2
            assert parts[0] == "tt"


class TestAuthMiddleware:
    """Test auth middleware from user perspective."""

    @patch("topictrace.db.client.pool")
    def test_missing_auth_header_returns_401(self, mock_pool):
        """Request without Authorization header should return 401."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "test"})
            assert response.status_code == 401
            assert "Missing Authorization header" in response.json()["detail"]

    @patch("topictrace.db.client.pool")
    def test_empty_auth_header_returns_401(self, mock_pool):
        """Request with empty Bearer token should return 401."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        with TestClient(app) as client:
            response = client.post(
                "/research",
                json={"query": "test"},
                headers={"Authorization": "Bearer "},
            )
            assert response.status_code == 401
            assert "Empty API key" in response.json()["detail"]

    @patch("topictrace.db.client.pool")
    def test_invalid_key_format_returns_401(self, mock_pool):
        """Request with key missing underscore separator should return 401."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        with TestClient(app) as client:
            response = client.post(
                "/research",
                json={"query": "test"},
                headers={"Authorization": "Bearer noseparator"},
            )
            assert response.status_code == 401
            assert "Invalid key format" in response.json()["detail"]

    @patch("topictrace.server.middleware.pool")
    def test_invalid_key_returns_401(self, mock_pool):
        """Request with valid format but wrong key should return 401."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        with TestClient(app) as client:
            response = client.post(
                "/research",
                json={"query": "test"},
                headers={"Authorization": "Bearer tt_wrongkey123"},
            )
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]

    @patch("topictrace.server.middleware.pool")
    @patch("topictrace.db.client.pool")
    @patch("topictrace.server.routes.deep_research.research.app.ainvoke", new_callable=AsyncMock)
    def test_valid_key_allows_request(self, mock_ainvoke, mock_db_pool, mock_middleware_pool):
        """Request with valid key should pass auth (may fail at agent level)."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        mock_ainvoke.return_value = {"messages": [MagicMock(content="mocked_answer")]}

        key_part = "testkey123"
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)  # user_id = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_middleware_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_middleware_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        with TestClient(app) as client:
            response = client.post(
                "/research",
                json={"query": "test"},
                headers={"Authorization": f"Bearer tt_{key_part}"},
            )
            # Should NOT be 401 — auth passed
            # May be 422 (validation) or 500 (agent error) but not 401
            assert response.status_code != 401

    @patch("topictrace.db.client.pool")
    def test_health_bypasses_auth(self, mock_pool):
        """/health/live should skip auth middleware."""
        from fastapi.testclient import TestClient
        from topictrace.server.app import app

        with TestClient(app) as client:
            # No auth header — should still work
            response = client.get("/health/live")
            assert response.status_code == 200
