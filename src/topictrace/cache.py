"""Cache system for TopicTrace with 20-minute TTL."""

import json
import os
import time

from topictrace import settings


def create_cache_key(prefix: str, value: str) -> str:
    """Create a safe alphanumeric filename for a cache key using prefix and value."""
    import re
    safe_value = re.sub(r'[^a-zA-Z0-9]', '-', value)[:50]
    # Collapse multiple dashes
    safe_value = re.sub(r'-+', '-', safe_value).strip('-')
    return f"{prefix}_{safe_value}"


def _get_cache_file_path(session_path: str, cache_key: str) -> str:
    """Return the full file path for a cache key JSON file."""
    return os.path.join(session_path, "cache", f"{cache_key}.json")


def save_to_cache(session_path: str, cache_key: str, data: dict) -> None:
    """Save JSON-serializable data and current timestamp to the cache directory."""
    cache_file = _get_cache_file_path(session_path, cache_key)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache_content = {
        "data": data,
        "timestamp": time.time()
    }
    with open(cache_file, "w") as f:
        json.dump(cache_content, f, indent=2)


def load_from_cache(session_path: str, cache_key: str) -> dict | None:
    """Load and return cached data, or None if the cache file does not exist or is corrupt."""
    cache_file = _get_cache_file_path(session_path, cache_key)
    try:
        with open(cache_file, "r") as f:
            cache_content = json.load(f)
        return cache_content["data"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def is_cache_valid(session_path: str, cache_key: str) -> bool:
    """Return True if cache entry exists and is within TTL, False otherwise."""
    cache_file = _get_cache_file_path(session_path, cache_key)
    try:
        with open(cache_file, "r") as f:
            cache_content = json.load(f)
        cache_age = time.time() - cache_content["timestamp"]
        return cache_age < settings.CACHE_TTL_SECONDS
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False
