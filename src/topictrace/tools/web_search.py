"""Web search tool for TopicTrace using Tavily API."""

import asyncio

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from topictrace import log, settings


def _extract_single_result(item: dict) -> dict:
    """Extract clean result dict from a Tavily search result item."""
    return {
        "title": item.get("title", "No Title"),
        "url": item.get("url", ""),
        "snippet": item.get("content", "")[: settings.SEARCH_SNIPPET_MAX_CHARS],
    }


@tool
async def web_search(query: str | list[str]) -> list[dict]:
    """Search the web using Tavily API, save the results, and return them as a list of dicts.

    Args:
        query: A single query string or a list of query strings.

    Returns:
        List of dicts: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    if isinstance(query, str):
        query = [query]
    elif not isinstance(query, list):
        error_message = (
            f"query parameter must be str or list, got {type(query).__name__}"
        )
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

    try:
        async with AsyncTavilyClient(api_key=settings.TAVILY_API_KEY) as client:
            tasks = [
                client.search(query=q, max_results=settings.SEARCH_MAX_RESULTS)
                for q in query
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                log.warning("search_failed", query=query[i], error=str(response))
                search_results.append(
                    {"title": "", "url": "", "snippet": f"Request failed: {response}"}
                )
                continue

            results_list = response.get("results", [])
            if not results_list:
                log.warning("search_no_results", query=query[i])
                search_results.append(
                    {
                        "title": "",
                        "url": "",
                        "snippet": f"No results found for: {query[i]}",
                    }
                )
                continue

            for result_item in results_list:
                clean = _extract_single_result(result_item)
                search_results.append(clean)

            log.info("search_success", query=query[i], results_count=len(results_list))

    except Exception as e:
        log.error("unexpected_error", error=str(e))
        search_results.append(
            {"title": "", "url": "", "snippet": f"Unexpected error: {e}"}
        )
    finally:
        log.info(
            "search_batch_complete",
            total=len(query),
            success=sum(1 for r in search_results if r.get("url")),
        )

    return search_results
