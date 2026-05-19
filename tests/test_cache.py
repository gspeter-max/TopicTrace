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
