"""
Web search tool for TopicTrace using Tavily API.

Tavily is an AI-optimized search API that returns clean,
structured results perfect for research agents.
"""

import os
from tavily import TavilyClient


def web_search(query: str) -> list[dict]:
    """
    Search the web using Tavily API.

    Uses the Tavily API to search for information on the web.
    Returns clean, structured search results with title, url, and snippet.

    Args:
        query: The search query string (e.g., "A-Level Biology past papers 2024")

    Returns:
        List of dictionaries, each with keys:
            - title: The title of the search result
            - url: The URL of the search result
            - snippet: A brief snippet/summary of the content
    """
    # Get API key from environment variable
    api_key = os.getenv("TAVILY_API_KEY")

    # Create Tavily client and perform search
    client = TavilyClient(api_key=api_key)
    response = client.search(query)

    # Transform results to our standard format
    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")
        })

    return results
