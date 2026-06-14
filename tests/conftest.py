"""Test configuration — mock DB pool before any topictrace imports."""

import os
from unittest.mock import MagicMock, patch

# Set fake DATABASE_URL so pool doesn't try to reach real postgres
os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/fake"

from unittest.mock import Mock


class MockConnectionPoolMeta(type(MagicMock)):
    def __instancecheck__(cls, instance):
        if isinstance(instance, (Mock, MagicMock)):
            return True
        return super().__instancecheck__(instance)

class MockConnectionPool(MagicMock, metaclass=MockConnectionPoolMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __class_getitem__(cls, item):
        return cls

# Patch ConnectionPool class BEFORE db/client.py is imported by any test
# This prevents the real pool from trying to connect at import time
_pool_patch = patch("psycopg_pool.ConnectionPool", new=MockConnectionPool)
_pool_patch.start()


from typing import Any
from topictrace.rag.documentRetrieve.graph.state import RAGState


def make_state(**overrides: Any) -> RAGState:
    """
    Build a RAGState with sensible defaults, overriding only what the test cares about.

    RAGState is a Pydantic BaseModel — every field is required unless given a
    default, so we supply a full set of zero-value defaults here and let callers
    override only the fields relevant to the behaviour under test.
    """
    defaults: dict[str, Any] = {
        "query": "",
        "top_k": 3,
        "top_k_rerank": 3,
        "intent": "",
        "raw_chunks": [],
        "vector_texts": [],
        "grade_sufficient": False,
        "grade_reason": "",
        "grade_answer": "",
        "graph_facts": "",
        "used_graph_search": False,
        "reason_for_graph_search": "",
        "final_context": [],
        "answer": "",
    }
    defaults.update(overrides)
    return RAGState(**defaults)
