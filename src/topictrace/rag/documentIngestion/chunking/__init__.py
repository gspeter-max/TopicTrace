"""
Public API for document chunking.

Import from here instead of individual strategy modules.
"""

from .recursive import chunk_document, count_tokens

__all__ = ["chunk_document", "count_tokens"]
