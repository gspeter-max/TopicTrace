"""
Web fetch tool for TopicTrace using Jina Reader.

Jina Reader converts any URL to clean Markdown.
Just prepend "https://r.jina.ai/" to any URL.
"""

import requests


def web_fetch(url: str) -> str:
    """
    Fetch a web page and return clean Markdown content using Jina Reader.

    Jina Reader (r.jina.ai) converts any web page to clean Markdown.
    No API key needed. Just prepend the URL with "https://r.jina.ai/".

    Args:
        url: The URL to fetch (e.g., "https://example.com/page")

    Returns:
        Clean Markdown content of the page, or empty string if fetch fails
    """
    # Build Jina Reader URL by prepending the proxy
    jina_url = f"https://r.jina.ai/{url}"

    try:
        response = requests.get(jina_url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception:
        # Return empty string if fetch fails for any reason
        return ""
