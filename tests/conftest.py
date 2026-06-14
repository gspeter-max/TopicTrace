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
