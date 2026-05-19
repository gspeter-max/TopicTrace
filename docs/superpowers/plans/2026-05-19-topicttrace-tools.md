# TopicTrace Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent:**
- Build two core tools for TopicTrace: `web_search` and `web_fetch`
- Plus a `summarize` tool that uses GLM-5.1 via NVIDIA NIM
- All tools must work with session folders for storage
- 20-minute TTL cache to avoid re-fetching
- Use `pyproject.toml` + `uv` for dependencies (NOT pip/requirements.txt)
- Follow TDD: write failing test first, then implement, then verify

**Files to read first:**
- `TopicTrace_Architecture.md` — full architecture decisions and tool flow
- `~/.openclaude/rules/coding.md` — coding rules (iron laws, no over-engineering)
- `~/.openclaude/rules/testing.md` — TDD mandate, test design principles

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    TopicTrace CLI                            │
├─────────────────────────────────────────────────────────────┤
│  web_search(query)                                          │
│  └─ Tavily API only (no fallback)                           │
│  Returns: List of {title, url, snippet}                     │
├─────────────────────────────────────────────────────────────┤
│  web_fetch(url)                                             │
│  └─ Jina Reader only (r.jina.ai/URL, no fallback)          │
│  Returns: Clean Markdown content                            │
├─────────────────────────────────────────────────────────────┤
│  summarize(content, query)                                  │
│  └─ NVIDIA NIM GLM-5.1 via openai client                   │
│  Returns: Concise summary string                            │
├─────────────────────────────────────────────────────────────┤
│  Cache Layer                                                │
│  └─ 20-minute TTL, JSON files in session/cache/             │
├─────────────────────────────────────────────────────────────┤
│  Session Folders                                            │
│  └─ sessions/<session-name>/                                │
│     search_results.md, fetched_pages/, summaries/, cache/   │
└─────────────────────────────────────────────────────────────┘
```

**Important Rules to follow:**
- **CRITICAL:** Add detailed docs in functions and explain the code and logic in comments
- **CRITICAL:** Make function names and variable names clear and literal — a 5-year-old child should understand
- Do NOT put any imagination or analogy in names — write code for developer highest speed to read
- Explain like a fresher — write docs in step-by-step simple style
- Make documentation human-readable and literal
- NO over-engineering: simplest solution that works is the best
- NO fallbacks: pick ONE tool per layer, master it
- TDD: write failing test FIRST, then implement

**Note on session_path:**
- `session_path` is created by `create_session()` in Task 3
- Every tool function receives `session_path` as a parameter
- The CLI entry point (future work) will call `create_session()` once, then pass the path to all tools
- This plan covers tools only — the agent loop and CLI are separate future tasks

---

## Task 1: Read instruction files

- [ ] **Step 1: Read CLAUDE.md and coding rules**

Read these files to understand the coding standards:
- `~/.openclaude/rules/coding.md`
- `~/.openclaude/rules/testing.md`
- `~/.openclaude/rules/completion.md`

- [ ] **Step 2: Read the architecture doc**

Read `TopicTrace_Architecture.md` to understand the full system design.

---

## Task 2: Project Setup with pyproject.toml and uv

**Files:**
- Create: `pyproject.toml`
- Create: `src/topictrace/__init__.py`
- Create: `src/topictrace/tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/tools/__init__.py`

- [ ] **Step 1: Write the failing test for project structure**

```python
# tests/test_project_structure.py
def test_package_importable():
    """Test that the topictrace package can be imported."""
    import topictrace
    assert topictrace is not None

def test_tools_subpackage_importable():
    """Test that the tools subpackage can be imported."""
    from topictrace import tools
    assert tools is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_project_structure.py -v`
Expected: FAIL with "No module named 'topictrace'"

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "topictrace"
version = "0.1.0"
description = "Educational predictive analytics agent for exam prep"
requires-python = ">=3.10"
dependencies = [
    "tavily-python>=0.5.0",
    "requests>=2.31.0",
    "openai>=1.50.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "prompt-toolkit>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/topictrace"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Create package structure and .env.example**

Create these files (can be empty `__init__.py`):
- `src/topictrace/__init__.py`
- `src/topictrace/tools/__init__.py`
- `tests/__init__.py`
- `tests/tools/__init__.py`

Create `.env.example` as a template for API keys:
```
# TopicTrace Environment Variables
# Copy this file to .env and fill in your API keys

# Tavily API key (get free key at https://tavily.com)
TAVILY_API_KEY=your-tavily-api-key-here

# NVIDIA API key (get free key at https://build.nvidia.com)
NVIDIA_API_KEY=your-nvidia-api-key-here
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_project_structure.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: initialize project with pyproject.toml and uv"
```

---

## Task 3: Session Folder Manager

**Files:**
- Create: `src/topictrace/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write the failing tests for session management**

```python
# tests/test_session.py
import os
import shutil
from topictrace.session import create_session, get_session_path


def test_create_session_creates_directory():
    """Test that create_session creates a session directory."""
    session_name = "test-session-create"
    session_path = create_session(session_name)
    assert os.path.isdir(session_path)
    # Cleanup
    shutil.rmtree(session_path)


def test_create_session_creates_subdirectories():
    """Test that create_session creates required subdirectories."""
    session_name = "test-session-subdirs"
    session_path = create_session(session_name)
    assert os.path.isdir(os.path.join(session_path, "fetched_pages"))
    assert os.path.isdir(os.path.join(session_path, "summaries"))
    assert os.path.isdir(os.path.join(session_path, "cache"))
    # Cleanup
    shutil.rmtree(session_path)


def test_get_session_path_returns_correct_path():
    """Test that get_session_path returns the correct path for a session name."""
    session_name = "test-session-path"
    expected_path = os.path.join("sessions", session_name)
    actual_path = get_session_path(session_name)
    assert actual_path == expected_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL with "No module named 'topictrace.session'"

- [ ] **Step 3: Implement session manager**

```python
# src/topictrace/session.py
"""
Session folder manager for TopicTrace.

Each research query gets its own isolated folder.
This keeps data organized and prevents context pollution.
"""

import os

# Base directory for all sessions
SESSIONS_DIR = "sessions"


def get_session_path(session_name: str) -> str:
    """
    Get the full path for a session folder.

    Args:
        session_name: Name of the session (e.g., "A-Level-Biology-2024")

    Returns:
        Full path to the session directory (e.g., "sessions/A-Level-Biology-2024")
    """
    return os.path.join(SESSIONS_DIR, session_name)


def create_session(session_name: str) -> str:
    """
    Create a new session folder with all required subdirectories.

    Creates:
        sessions/<session_name>/
        sessions/<session_name>/fetched_pages/
        sessions/<session_name>/summaries/
        sessions/<session_name>/cache/

    Args:
        session_name: Name of the session (e.g., "A-Level-Biology-2024")

    Returns:
        Full path to the created session directory
    """
    session_path = get_session_path(session_name)

    # Create main session directory
    os.makedirs(session_path, exist_ok=True)

    # Create subdirectories for different data types
    os.makedirs(os.path.join(session_path, "fetched_pages"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "cache"), exist_ok=True)

    return session_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/session.py tests/test_session.py
git commit -m "feat: add session folder manager"
```

---

## Task 4: Cache System (20-minute TTL)

**Files:**
- Create: `src/topictrace/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the failing tests for cache system**

```python
# tests/test_cache.py
import os
import json
import shutil
import time
from topictrace.session import create_session
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid


def test_save_to_cache_creates_file():
    """Test that save_to_cache creates a JSON file in the cache directory."""
    session_name = "test-cache-save"
    session_path = create_session(session_name)
    cache_key = "test-key"
    data = {"title": "Test", "url": "https://example.com"}

    save_to_cache(session_path, cache_key, data)

    cache_file = os.path.join(session_path, "cache", f"{cache_key}.json")
    assert os.path.exists(cache_file)

    # Cleanup
    shutil.rmtree(session_path)


def test_load_from_cache_returns_saved_data():
    """Test that load_from_cache returns the data that was saved."""
    session_name = "test-cache-load"
    session_path = create_session(session_name)
    cache_key = "test-key"
    data = {"title": "Test", "url": "https://example.com"}

    save_to_cache(session_path, cache_key, data)
    loaded = load_from_cache(session_path, cache_key)

    assert loaded == data

    # Cleanup
    shutil.rmtree(session_path)


def test_is_cache_valid_returns_true_for_fresh_cache():
    """Test that is_cache_valid returns True for cache less than 20 minutes old."""
    session_name = "test-cache-valid"
    session_path = create_session(session_name)
    cache_key = "test-key"
    data = {"title": "Test"}

    save_to_cache(session_path, cache_key, data)

    assert is_cache_valid(session_path, cache_key) is True

    # Cleanup
    shutil.rmtree(session_path)


def test_is_cache_valid_returns_false_for_expired_cache():
    """Test that is_cache_valid returns False for cache older than 20 minutes."""
    session_name = "test-cache-expired"
    session_path = create_session(session_name)
    cache_key = "test-key"
    data = {"title": "Test"}

    # Save cache with old timestamp (21 minutes ago)
    cache_file = os.path.join(session_path, "cache", f"{cache_key}.json")
    old_time = time.time() - (21 * 60)  # 21 minutes ago
    cache_content = {
        "data": data,
        "timestamp": old_time
    }
    with open(cache_file, "w") as f:
        json.dump(cache_content, f)

    assert is_cache_valid(session_path, cache_key) is False

    # Cleanup
    shutil.rmtree(session_path)


def test_load_from_cache_returns_none_for_missing_key():
    """Test that load_from_cache returns None when key doesn't exist."""
    session_name = "test-cache-missing"
    session_path = create_session(session_name)

    loaded = load_from_cache(session_path, "nonexistent-key")

    assert loaded is None

    # Cleanup
    shutil.rmtree(session_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with "No module named 'topictrace.cache'"

- [ ] **Step 3: Implement cache system**

```python
# src/topictrace/cache.py
"""
Simple file-based cache with 20-minute TTL (Time To Live).

Saves data as JSON files in the session's cache directory.
Each cache entry includes a timestamp to check if it's still fresh.
"""

import json
import os
import time

# Cache expires after 20 minutes (in seconds)
CACHE_TTL_SECONDS = 20 * 60


def _get_cache_file_path(session_path: str, cache_key: str) -> str:
    """
    Get the full path to a cache file.

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry (e.g., "search_exam-prep")

    Returns:
        Full path to the cache JSON file
    """
    return os.path.join(session_path, "cache", f"{cache_key}.json")


def save_to_cache(session_path: str, cache_key: str, data: any) -> None:
    """
    Save data to cache with current timestamp.

    Creates a JSON file with structure:
        {
            "data": <your data>,
            "timestamp": <current unix timestamp>
        }

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry
        data: Any JSON-serializable data to cache
    """
    cache_file = _get_cache_file_path(session_path, cache_key)

    cache_content = {
        "data": data,
        "timestamp": time.time()
    }

    with open(cache_file, "w") as f:
        json.dump(cache_content, f, indent=2)


def load_from_cache(session_path: str, cache_key: str) -> any:
    """
    Load data from cache if the file exists.

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry

    Returns:
        The cached data, or None if the cache file doesn't exist
    """
    cache_file = _get_cache_file_path(session_path, cache_key)

    if not os.path.exists(cache_file):
        return None

    with open(cache_file, "r") as f:
        cache_content = json.load(f)

    return cache_content.get("data")


def is_cache_valid(session_path: str, cache_key: str) -> bool:
    """
    Check if cached data is still fresh (less than 20 minutes old).

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry

    Returns:
        True if cache exists and is less than 20 minutes old, False otherwise
    """
    cache_file = _get_cache_file_path(session_path, cache_key)

    if not os.path.exists(cache_file):
        return False

    with open(cache_file, "r") as f:
        cache_content = json.load(f)

    timestamp = cache_content.get("timestamp", 0)
    age_seconds = time.time() - timestamp

    return age_seconds < CACHE_TTL_SECONDS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/cache.py tests/test_cache.py
git commit -m "feat: add cache system with 20-minute TTL"
```

---

## Task 5: web_search Tool (Tavily)

**Files:**
- Create: `src/topictrace/tools/web_search.py`
- Create: `tests/tools/test_web_search.py`

- [ ] **Step 1: Write the failing tests for web_search**

```python
# tests/tools/test_web_search.py
import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.web_search import web_search


def test_web_search_returns_list_of_results():
    """Test that web_search returns a list with title, url, snippet."""
    session_name = "test-web-search-results"
    session_path = create_session(session_name)

    # Mock Tavily client to avoid real API calls
    mock_response = {
        "results": [
            {
                "title": "AQA Biology Past Papers",
                "url": "https://example.com/aqa-biology",
                "content": "Past papers for AQA A-Level Biology..."
            },
            {
                "title": "OCR Biology Syllabus",
                "url": "https://example.com/ocr-biology",
                "content": "The OCR Biology specification covers..."
            }
        ]
    }

    with patch("topictrace.tools.web_search.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = mock_response
        results = web_search("A-Level Biology exam prep", session_path)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["title"] == "AQA Biology Past Papers"
    assert results[0]["url"] == "https://example.com/aqa-biology"
    assert "snippet" in results[0]

    # Cleanup
    shutil.rmtree(session_path)


def test_web_search_saves_results_to_file():
    """Test that web_search saves results to search_results.md in session folder."""
    session_name = "test-web-search-save"
    session_path = create_session(session_name)

    mock_response = {
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "content": "Test content snippet"
            }
        ]
    }

    with patch("topictrace.tools.web_search.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = mock_response
        web_search("test query", session_path)

    results_file = os.path.join(session_path, "search_results.md")
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        content = f.read()

    assert "Test Title" in content
    assert "https://example.com" in content

    # Cleanup
    shutil.rmtree(session_path)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_web_search.py -v`
Expected: FAIL with "No module named 'topictrace.tools.web_search'"

- [ ] **Step 3: Implement web_search tool**

```python
# src/topictrace/tools/web_search.py
"""
Web search tool using Tavily API.

Searches the web for exam-related content like past papers,
syllabi, and study materials. Returns structured results
with title, URL, and content snippet.

No fallback — Tavily is the only search provider.
"""

import os
from tavily import TavilyClient
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid


def _create_cache_key(query: str) -> str:
    """
    Create a safe cache key from a search query.

    Converts query to lowercase and replaces spaces with dashes.
    Example: "A-Level Biology" → "search_a-level-biology"

    Args:
        query: The search query string

    Returns:
        A safe filename string for caching
    """
    safe_query = query.lower().replace(" ", "-")[:50]
    return f"search_{safe_query}"


def _save_results_to_file(results: list, session_path: str) -> None:
    """
    Save search results as a Markdown file in the session folder.

    Creates a file like:
        # Search Results
        ## 1. Title Here
        - URL: https://example.com
        - Snippet: Content preview...

    Args:
        results: List of result dicts with title, url, snippet
        session_path: Path to the session directory
    """
    results_file = os.path.join(session_path, "search_results.md")

    with open(results_file, "w") as f:
        f.write("# Search Results\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"## {i}. {result['title']}\n")
            f.write(f"- URL: {result['url']}\n")
            f.write(f"- Snippet: {result['snippet']}\n\n")


def web_search(query: str, session_path: str) -> list:
    """
    Search the web using Tavily API.

    Flow:
        1. Check cache → return cached results if fresh (less than 20 min)
        2. Call Tavily API with the query
        3. Extract title, url, snippet from each result
        4. Save results to session/search_results.md
        5. Cache results for future use

    Args:
        query: Search query string (e.g., "A-Level Biology past papers")
        session_path: Path to the session directory for saving results

    Returns:
        List of dicts, each with:
            - title: Page title
            - url: Page URL
            - snippet: Short content preview

    Raises:
        ValueError: If TAVILY_API_KEY is not set in environment
    """
    # Step 1: Check cache
    cache_key = _create_cache_key(query)
    if is_cache_valid(session_path, cache_key):
        cached = load_from_cache(session_path, cache_key)
        if cached is not None:
            return cached

    # Step 2: Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Set it in your .env file: TAVILY_API_KEY=your-key-here"
        )

    # Step 3: Call Tavily API
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=10)

    # Step 4: Extract clean results
    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", "No Title"),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:300]  # Limit snippet length
        })

    # Step 5: Save to file and cache
    _save_results_to_file(results, session_path)
    save_to_cache(session_path, cache_key, results)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_web_search.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/tools/web_search.py tests/tools/test_web_search.py
git commit -m "feat: add web_search tool using Tavily API"
```

---

## Task 6: web_fetch Tool (Jina Reader)

**Files:**
- Create: `src/topictrace/tools/web_fetch.py`
- Create: `tests/tools/test_web_fetch.py`

- [ ] **Step 1: Write the failing tests for web_fetch**

```python
# tests/tools/test_web_fetch.py
import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.web_fetch import web_fetch


def test_web_fetch_returns_markdown_content():
    """Test that web_fetch returns clean Markdown content from Jina Reader."""
    session_name = "test-web-fetch-content"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.text = "# Example Page\n\nThis is the page content in Markdown."
    mock_response.status_code = 200

    with patch("topictrace.tools.web_fetch.requests.get") as mock_get:
        mock_get.return_value = mock_response
        content = web_fetch("https://example.com", session_path)

    assert "# Example Page" in content
    assert "page content in Markdown" in content

    # Cleanup
    shutil.rmtree(session_path)


def test_web_fetch_saves_content_to_file():
    """Test that web_fetch saves fetched content to fetched_pages/ directory."""
    session_name = "test-web-fetch-save"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.text = "# Test Content\n\nSaved to file."
    mock_response.status_code = 200

    with patch("topictrace.tools.web_fetch.requests.get") as mock_get:
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

    # Cleanup
    shutil.rmtree(session_path)


def test_web_fetch_uses_cache_when_valid():
    """Test that web_fetch returns cached content when cache is fresh."""
    session_name = "test-web-fetch-cache"
    session_path = create_session(session_name)

    # Pre-populate cache
    from topictrace.cache import save_to_cache
    cached_content = "# Cached Page\n\nThis is cached content."
    cache_key = "fetch_https---example-com"
    save_to_cache(session_path, cache_key, cached_content)

    # Should return cached content without making HTTP request
    content = web_fetch("https://example.com", session_path)

    assert content == cached_content

    # Cleanup
    shutil.rmtree(session_path)


def test_web_fetch_raises_on_http_error():
    """Test that web_fetch raises an exception when Jina returns an error."""
    session_name = "test-web-fetch-error"
    session_path = create_session(session_name)

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("topictrace.tools.web_fetch.requests.get") as mock_get:
        mock_get.return_value = mock_response
        try:
            web_fetch("https://example.com", session_path)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "500" in str(e) or "error" in str(e).lower()

    # Cleanup
    shutil.rmtree(session_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_web_fetch.py -v`
Expected: FAIL with "No module named 'topictrace.tools.web_fetch'"

- [ ] **Step 3: Implement web_fetch tool**

```python
# src/topictrace/tools/web_fetch.py
"""
Web fetch tool using Jina Reader API.

Converts any URL to clean Markdown by calling Jina Reader
at r.jina.ai. No browser automation needed — just a simple
HTTP GET request.

No fallback — Jina Reader is the only fetch provider.
"""

import os
import re
import requests
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid


# Jina Reader base URL — prepend to any URL to get Markdown
JINA_READER_BASE_URL = "https://r.jina.ai/"


def _create_cache_key(url: str) -> str:
    """
    Create a safe cache key from a URL.

    Replaces special characters with dashes.
    Example: "https://example.com/page" → "fetch_https---example-com-page"

    Args:
        url: The URL to fetch

    Returns:
        A safe filename string for caching
    """
    safe_url = re.sub(r'[^a-zA-Z0-9]', '-', url)[:80]
    return f"fetch_{safe_url}"


def _save_content_to_file(content: str, url: str, session_path: str) -> None:
    """
    Save fetched Markdown content to the fetched_pages directory.

    Creates a numbered file like page_1.md, page_2.md, etc.

    Args:
        content: The Markdown content to save
        url: The source URL (saved as a comment in the file)
        session_path: Path to the session directory
    """
    fetched_dir = os.path.join(session_path, "fetched_pages")

    # Count existing files to determine the next number
    existing_files = [f for f in os.listdir(fetched_dir) if f.endswith(".md")]
    next_number = len(existing_files) + 1

    filename = f"page_{next_number}.md"
    filepath = os.path.join(fetched_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"<!-- Source: {url} -->\n\n")
        f.write(content)


def web_fetch(url: str, session_path: str) -> str:
    """
    Fetch a web page and convert it to clean Markdown.

    Flow:
        1. Check cache → return cached content if fresh (less than 20 min)
        2. Call Jina Reader API (r.jina.ai/URL)
        3. Check response status — raise on error
        4. Save content to session/fetched_pages/page_N.md
        5. Cache content for future use

    Args:
        url: The URL to fetch (e.g., "https://example.com/page")
        session_path: Path to the session directory for saving content

    Returns:
        Clean Markdown content as a string

    Raises:
        Exception: If Jina Reader returns a non-200 status code
    """
    # Step 1: Check cache
    cache_key = _create_cache_key(url)
    if is_cache_valid(session_path, cache_key):
        cached = load_from_cache(session_path, cache_key)
        if cached is not None:
            return cached

    # Step 2: Call Jina Reader API
    jina_url = f"{JINA_READER_BASE_URL}{url}"
    response = requests.get(jina_url, timeout=30)

    # Step 3: Check for errors
    if response.status_code != 200:
        raise Exception(
            f"Jina Reader returned status {response.status_code} "
            f"for URL: {url}"
        )

    content = response.text

    # Step 4: Save to file and cache
    _save_content_to_file(content, url, session_path)
    save_to_cache(session_path, cache_key, content)

    return content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_web_fetch.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/tools/web_fetch.py tests/tools/test_web_fetch.py
git commit -m "feat: add web_fetch tool using Jina Reader"
```

---

## Task 7: summarize Tool (GLM-5.1 via NVIDIA NIM)

**Files:**
- Create: `src/topictrace/tools/summarize.py`
- Create: `tests/tools/test_summarize.py`

- [ ] **Step 1: Write the failing tests for summarize**

```python
# tests/tools/test_summarize.py
import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.summarize import summarize


def test_summarize_returns_summary_string():
    """Test that summarize returns a summary string from GLM-5.1."""
    session_name = "test-summarize-return"
    session_path = create_session(session_name)

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "This is a summary of the content."
    mock_chunk.choices[0].delta.reasoning_content = None

    with patch("topictrace.tools.summarize.client") as mock_client:
        mock_client.chat.completions.create.return_value = [mock_chunk]
        result = summarize(
            "Long content about biology...",
            "What are the key topics?",
            session_path
        )

    assert isinstance(result, str)
    assert "summary" in result.lower()

    # Cleanup
    shutil.rmtree(session_path)


def test_summarize_saves_to_summaries_directory():
    """Test that summarize saves the summary to summaries/ directory."""
    session_name = "test-summarize-save"
    session_path = create_session(session_name)

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Summary saved."
    mock_chunk.choices[0].delta.reasoning_content = None

    with patch("topictrace.tools.summarize.client") as mock_client:
        mock_client.chat.completions.create.return_value = [mock_chunk]
        summarize("content", "query", session_path)

    summaries_dir = os.path.join(session_path, "summaries")
    files = os.listdir(summaries_dir)
    assert len(files) >= 1

    # Cleanup
    shutil.rmtree(session_path)


def test_summarize_raises_on_empty_content():
    """Test that summarize raises ValueError when content is empty."""
    session_name = "test-summarize-empty"
    session_path = create_session(session_name)

    try:
        summarize("", "query", session_path)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Cleanup
    shutil.rmtree(session_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_summarize.py -v`
Expected: FAIL with "No module named 'topictrace.tools.summarize'"

- [ ] **Step 3: Implement summarize tool**

```python
# src/topictrace/tools/summarize.py
"""
Summarization tool using GLM-5.1 via NVIDIA NIM.

Takes long content (from web_fetch) and a query (from user),
then uses GLM-5.1 to produce a concise, relevant summary.
This keeps the agent's context window clean.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client pointing to NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Model to use for summarization
MODEL_NAME = "z-ai/glm-5.1"


def _save_summary_to_file(summary: str, session_path: str) -> None:
    """
    Save summary to the summaries directory.

    Creates a numbered file like summary_1.md, summary_2.md, etc.

    Args:
        summary: The summary text to save
        session_path: Path to the session directory
    """
    summaries_dir = os.path.join(session_path, "summaries")

    # Count existing files to determine the next number
    existing_files = [f for f in os.listdir(summaries_dir) if f.endswith(".md")]
    next_number = len(existing_files) + 1

    filename = f"summary_{next_number}.md"
    filepath = os.path.join(summaries_dir, filename)

    with open(filepath, "w") as f:
        f.write(summary)


def summarize(content: str, query: str, session_path: str) -> str:
    """
    Summarize content using GLM-5.1 based on the user's query.

    Sends the content and query to GLM-5.1 via NVIDIA NIM.
    The model produces a concise summary relevant to the query.

    Args:
        content: The full text to summarize (from web_fetch)
        query: The user's original question (for context)
        session_path: Path to the session directory for saving

    Returns:
        A concise summary string

    Raises:
        ValueError: If content is empty
        Exception: If NVIDIA NIM API call fails
    """
    # Validate input
    if not content or not content.strip():
        raise ValueError("Content cannot be empty for summarization")

    # Build the prompt for GLM-5.1
    messages = [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. "
                "Summarize the provided content in relation to the user's query. "
                "Be concise, factual, and focus on exam-relevant information. "
                "Output only the summary, no preamble."
            )
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Content to summarize:\n{content[:8000]}"  # Limit to avoid token overflow
            )
        }
    ]

    # Call GLM-5.1 via NVIDIA NIM
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )

    # Collect streamed response
    summary_parts = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None) is not None:
            summary_parts.append(delta.content)

    summary = "".join(summary_parts)

    # Save summary to file
    _save_summary_to_file(summary, session_path)

    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_summarize.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/tools/summarize.py tests/tools/test_summarize.py
git commit -m "feat: add summarize tool using GLM-5.1 via NVIDIA NIM"
```

---

## Task 8: Tool Registry (Wire Tools Together)

**Files:**
- Create: `src/topictrace/tools/registry.py`
- Create: `tests/tools/test_registry.py`

- [ ] **Step 1: Write the failing tests for tool registry**

```python
# tests/tools/test_registry.py
from topictrace.tools.registry import get_tool_definitions, execute_tool


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_registry.py -v`
Expected: FAIL with "No module named 'topictrace.tools.registry'"

- [ ] **Step 3: Implement tool registry**

```python
# src/topictrace/tools/registry.py
"""
Tool registry for TopicTrace.

Defines the tools available to the GLM-5.1 agent.
These definitions are passed to the openai client's
tools parameter for native JSON tool calling.
"""

from topictrace.tools.web_search import web_search
from topictrace.tools.web_fetch import web_fetch
from topictrace.tools.summarize import summarize


def get_tool_definitions() -> list:
    """
    Get the tool definitions for GLM-5.1 tool calling.

    Returns a list of tool definitions in OpenAI-compatible format.
    These tell the LLM what tools are available and what parameters
    each tool expects.

    Returns:
        List of tool definition dicts
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for exam-related content like past papers, "
                    "syllabi, study materials, and exam tips. "
                    "Returns a list of results with title, URL, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'A-Level Biology past papers')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "Fetch a web page and convert it to clean Markdown. "
                    "Use this after web_search to get the full content of a result. "
                    "Returns the page content as Markdown text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (e.g., 'https://example.com/page')"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": (
                    "Summarize long content in relation to the user's query. "
                    "Use this after web_fetch to condense page content "
                    "into exam-relevant highlights."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The full text content to summarize"
                        },
                        "query": {
                            "type": "string",
                            "description": "The user's original question for context"
                        }
                    },
                    "required": ["content", "query"]
                }
            }
        }
    ]


def execute_tool(tool_name: str, arguments: dict, session_path: str) -> str:
    """
    Execute a tool by name with the given arguments.

    Routes the tool call to the correct function based on tool_name.
    This is called when GLM-5.1 decides to use a tool.

    Args:
        tool_name: Name of the tool to execute ("web_search", "web_fetch", "summarize")
        arguments: Dict of arguments to pass to the tool
        session_path: Path to the current session directory

    Returns:
        The tool's output as a string

    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name == "web_search":
        results = web_search(arguments["query"], session_path)
        # Convert results list to string for the LLM
        return "\n".join(
            f"{i+1}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        )
    elif tool_name == "web_fetch":
        return web_fetch(arguments["url"], session_path)
    elif tool_name == "summarize":
        return summarize(arguments["content"], arguments["query"], session_path)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_registry.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add src/topictrace/tools/registry.py tests/tools/test_registry.py
git commit -m "feat: add tool registry for GLM-5.1 tool calling"
```

---

## Task 9: Integration Test — Full Tool Chain

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test for full tool chain**

```python
# tests/test_integration.py
import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.web_search import web_search
from topictrace.tools.web_fetch import web_fetch
from topictrace.tools.summarize import summarize


def test_full_tool_chain_search_fetch_summarize():
    """Test the full chain: search → fetch → summarize."""
    session_name = "test-integration-full"
    session_path = create_session(session_name)

    # Mock search results
    mock_search_response = {
        "results": [
            {
                "title": "AQA Biology Past Papers",
                "url": "https://example.com/aqa-biology",
                "content": "Past papers for AQA A-Level Biology..."
            }
        ]
    }

    # Mock fetch response
    mock_fetch_response = MagicMock()
    mock_fetch_response.text = "# AQA Biology\n\nCell biology is a key topic."
    mock_fetch_response.status_code = 200

    # Mock summarize response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Cell biology covers cell structure and function."
    mock_chunk.choices[0].delta.reasoning_content = None

    # Run the full chain
    with patch("topictrace.tools.web_search.TavilyClient") as MockTavily:
        MockTavily.return_value.search.return_value = mock_search_response
        search_results = web_search("AQA Biology", session_path)

    with patch("topictrace.tools.web_fetch.requests.get") as mock_get:
        mock_get.return_value = mock_fetch_response
        page_content = web_fetch(search_results[0]["url"], session_path)

    with patch("topictrace.tools.summarize.client") as mock_client:
        mock_client.chat.completions.create.return_value = [mock_chunk]
        summary = summarize(page_content, "AQA Biology", session_path)

    # Verify all outputs exist
    assert len(search_results) == 1
    assert "AQA Biology" in page_content
    assert "cell biology" in summary.lower()

    # Verify files were created
    assert os.path.exists(os.path.join(session_path, "search_results.md"))
    assert len(os.listdir(os.path.join(session_path, "fetched_pages"))) >= 1
    assert len(os.listdir(os.path.join(session_path, "summaries"))) >= 1

    # Cleanup
    shutil.rmtree(session_path)
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full tool chain"
```

---

## Task 10: Final Cleanup and Verification

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --cov=topictrace --cov-report=term-missing`
Expected: All tests PASS, coverage report shown

- [ ] **Step 2: Verify no TODOs left in code**

Run: `grep -r "TODO" src/`
Expected: No results

- [ ] **Step 3: Verify all imports work**

Run: `uv run python -c "from topictrace.tools import web_search, web_fetch, summarize; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```

---

## Self-Review Checklist

- [ ] All tests pass
- [ ] No TODOs in code
- [ ] Function names are clear and literal
- [ ] Every function has detailed docstrings
- [ ] No fallbacks — one tool per layer
- [ ] Cache uses 20-minute TTL
- [ ] Session folders created with correct structure
- [ ] pyproject.toml uses uv-compatible format
- [ ] No pip or requirements.txt references
- [ ] Integration test covers full tool chain
