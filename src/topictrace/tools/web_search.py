"""Web search tool for TopicTrace using Tavily API."""

import asyncio
from tavily import AsyncTavilyClient
from topictrace import settings, log
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid, create_cache_key
from langchain_core.tools import tool 

def _save_results_to_file(results: list, session_path: str) -> None:
    """Save search results as a Markdown file in the session directory."""
    import os
    os.makedirs(session_path, exist_ok=True)
    results_file = os.path.join(session_path, "search_results.md")

    with open(results_file, "w") as f:
        f.write("# Search Results\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"## {i}. {result['title']}\n")
            f.write(f"- URL: {result['url']}\n")
            f.write(f"- Snippet: {result['snippet']}\n\n")


def _extract_single_result(item: dict) -> dict:
    """Extract clean result dict from a Tavily search result item."""
    return {
        "title": item.get("title", "No Title"),
        "url": item.get("url", ""),
        "snippet": item.get("content", "")[:settings.SEARCH_SNIPPET_MAX_CHARS],
    }

@tool
async def web_search(query: str | list[str]) -> list[dict]:
    """Search the web using Tavily API, save the results, and return them as a list of dicts.

    Args:
        query: A single query string or a list of query strings.

    Returns:
        List of dicts: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    from topictrace.session import create_session
    session_path = create_session(query[:50] if isinstance(query, str) else query[0][:50])

    if isinstance(query, str):
        query = [query]
    elif not isinstance(query, list):
        error_message = f"query parameter must be str or list, got {type(query).__name__}"
        log.warning("invalid_input", error=error_message)
        return [{"title": "", "url": "", "snippet": error_message}]

    if not settings.TAVILY_API_KEY:
        error_message = (
            "TAVILY_API_KEY not found. "
            "Set it in your .env file: TAVILY_API_KEY=your-key-here"
        )
        log.warning(error_message)
        return [{"title": "", "url": "", "snippet": error_message}]

    search_results = []
    uncached_queries = []

    try:
        for q in query:
            cache_key = create_cache_key("search", q)
            if is_cache_valid(session_path, cache_key):
                cached = load_from_cache(session_path, cache_key)
                if cached is not None:
                    log.info("cache_hit", query=q)
                    if isinstance(cached, list):
                        search_results.extend(cached)
                    else:
                        search_results.append(_extract_single_result(cached))
                    continue
            uncached_queries.append({"query": q, "cache_key": cache_key})

        if not uncached_queries:
            return search_results

        async with AsyncTavilyClient(api_key=settings.TAVILY_API_KEY) as client:
            tasks = [
                client.search(query=item["query"], max_results=settings.SEARCH_MAX_RESULTS)
                for item in uncached_queries
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for item, response in zip(uncached_queries, responses):
            q = item["query"]

            if isinstance(response, Exception):
                log.warning("search_failed", query=q, error=str(response))
                search_results.append({"title": "", "url": "", "snippet": f"Request failed: {response}"})
                continue

            results_list = response.get("results", [])
            if not results_list:
                log.warning("search_no_results", query=q)
                search_results.append({"title": "", "url": "", "snippet": f"No results found for: {q}"})
                continue

            clean_results = []
            for result_item in results_list:
                clean = _extract_single_result(result_item)
                clean_results.append(clean)
                search_results.append(clean)

            _save_results_to_file(clean_results, session_path)
            save_to_cache(session_path, item["cache_key"], clean_results)
            log.info("search_success", query=q, results_count=len(results_list))

    except Exception as e:
        log.error("unexpected_error", error=str(e))
        search_results.append({"title": "", "url": "", "snippet": f"Unexpected error: {e}"})
    finally:
        log.info(
            "search_batch_complete",
            total=len(query),
            success=sum(1 for r in search_results if r.get("url")),
        )

    return search_results
