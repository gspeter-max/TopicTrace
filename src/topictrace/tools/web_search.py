"""
Web search tool for TopicTrace using Tavily API.

Tavily is an AI-optimized search API that returns clean,
structured results perfect for research agents.

No fallback — Tavily is the only search provider.
"""

from tavily import TavilyClient
from topictrace import settings
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid


def _create_cache_key(query: str) -> str:
    """
    Create a safe cache key from a search query.

    Converts query to lowercase and replaces spaces with dashes.
    Example: "A-Level Biology" → "search_a-level-biology"

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
    import os
    results_file = os.path.join(session_path, "search_results.md")

    with open(results_file, "w") as f:
        f.write("# Search Results\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"## {i}. {result['title']}\n")
            f.write(f"- URL: {result['url']}\n")
            f.write(f"- Snippet: {result['snippet']}\n\n")


def web_search(query: str, session_path: str) -> list[dict]:
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

    # Step 2: Get API key from settings — fail fast if missing
    if not settings.TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Set it in your .env file: TAVILY_API_KEY=your-key-here"
        )

    # Step 3: Call Tavily API
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = client.search(query=query, max_results=settings.SEARCH_MAX_RESULTS)

    # Step 4: Extract clean results
    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", "No Title"),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:settings.SEARCH_SNIPPET_MAX_CHARS]
        })

    # Step 5: Save to file and cache
    _save_results_to_file(results, session_path)
    save_to_cache(session_path, cache_key, results)

    return results
