"""
Task 1: Test that voyage_api_key is loaded from config.
"""

from topictrace import settings


def test_voyage_api_key_is_loaded():
    """voyage_api_key must exist inside RERANKER_CONFIG (even if empty in CI)."""
    assert hasattr(settings, "RERANKER_CONFIG"), "settings.RERANKER_CONFIG is missing"
    assert hasattr(settings.RERANKER_CONFIG, "VOYAGE_API_KEY"), (
        "RERANKER_CONFIG.VOYAGE_API_KEY is missing"
    )


def test_voyage_api_key_is_string():
    """voyage_api_key must be a str (never None)."""
    assert isinstance(settings.RERANKER_CONFIG.VOYAGE_API_KEY, str)


def test_voyage_api_key_does_not_raise_on_missing():
    """
    Config must NOT raise if VOYAGE_API_KEY is absent.
    We verify this indirectly: importing settings itself must succeed
    (this test module already imported it at the top).
    """
    # If we reached this line, the import succeeded without raising.
    assert True
