"""
Web fetch tool for TopicTrace using Jina Reader.

Jina Reader converts any URL to clean Markdown.
Just prepend "https://r.jina.ai/" to any URL.

No fallback — Jina Reader is the only fetch provider.
"""

import os
import requests
from topictrace import settings
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid, create_cache_key


def _save_content_to_file(content: str, url: str, session_path: str) -> None:
    """
    Save fetched Markdown content to the fetched_pages directory.

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

    Args:
        url: The URL to fetch
        session_path: Path to the session directory

    Returns:
        Clean Markdown content as a string

    Raises:
        ValueError: If url is empty
        Exception: If Jina Reader returns a non-200 status code
    """
    # Validate input
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    # Step 1: Check cache
    cache_key = create_cache_key("fetch", url)
    if is_cache_valid(session_path, cache_key):
        cached = load_from_cache(session_path, cache_key)
        if cached is not None:
            return cached

    # Step 2: Call Jina Reader API
    jina_url = f"{settings.JINA_READER_BASE_URL}{url}"
    response = requests.get(jina_url, timeout=settings.FETCH_TIMEOUT_SECONDS)

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
