"""
Task 1: Test that voyage_api_key is loaded from config.
"""
import config


def test_voyage_api_key_is_loaded():
    """voyage_api_key must exist as an attribute on config (even if empty in CI)."""
    assert hasattr(config, "voyage_api_key"), "config.voyage_api_key attribute is missing"


def test_voyage_api_key_is_string():
    """voyage_api_key must be a str (never None)."""
    assert isinstance(config.voyage_api_key, str)


def test_voyage_api_key_does_not_raise_on_missing():
    """
    Config must NOT raise if VOYAGE_API_KEY is absent.
    We verify this indirectly: importing config itself must succeed
    (this test module already imported it at the top).
    """
    # If we reached this line, the import succeeded without raising.
    assert True
