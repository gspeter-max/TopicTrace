"""
Cache system for TopicTrace with 20-minute TTL.

Caches search results and fetched pages to avoid re-fetching.
Each cache entry is a JSON file with data + timestamp.
"""

import json
import os
import time

from topictrace import settings


def _get_cache_file_path(session_path: str, cache_key: str) -> str:
    """
    Get the full path to a cache file.

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry (e.g., "search-python-tutoring")

    Returns:
        Full path to the cache JSON file
    """
    return os.path.join(session_path, "cache", f"{cache_key}.json")


def save_to_cache(session_path: str, cache_key: str, data: dict) -> None:
    """
    Save data to cache with current timestamp.

    Creates a JSON file with the data and the current time.
    This lets us check later if the cache is still fresh.

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry
        data: The data to cache (must be JSON-serializable)
    """
    cache_file = _get_cache_file_path(session_path, cache_key)
    cache_content = {
        "data": data,
        "timestamp": time.time()
    }
    with open(cache_file, "w") as f:
        json.dump(cache_content, f, indent=2)


def load_from_cache(session_path: str, cache_key: str) -> dict | None:
    """
    Load data from cache if the cache file exists.

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry

    Returns:
        The cached data, or None if cache file doesn't exist or is corrupted
    """
    cache_file = _get_cache_file_path(session_path, cache_key)
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r") as f:
            cache_content = json.load(f)
        return cache_content["data"]
    except (json.JSONDecodeError, KeyError):
        # Cache file is corrupted — treat as missing
        return None


def is_cache_valid(session_path: str, cache_key: str) -> bool:
    """
    Check if cache entry is still fresh (less than 20 minutes old).

    Args:
        session_path: Path to the session directory
        cache_key: Unique key for this cache entry

    Returns:
        True if cache exists and is less than 20 minutes old, False otherwise
    """
    cache_file = _get_cache_file_path(session_path, cache_key)
    if not os.path.exists(cache_file):
        return False
    try:
        with open(cache_file, "r") as f:
            cache_content = json.load(f)
        cache_age = time.time() - cache_content["timestamp"]
        return cache_age < settings.CACHE_TTL_SECONDS
    except (json.JSONDecodeError, KeyError):
        # Cache file is corrupted — treat as invalid
        return False
