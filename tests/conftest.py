"""Test configuration — mock DB pool before any topictrace imports."""

import os
from unittest.mock import MagicMock, patch

# Set fake DATABASE_URL so pool doesn't try to reach real postgres
os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/fake"

# Patch ConnectionPool class BEFORE db/client.py is imported by any test
# This prevents the real pool from trying to connect at import time
_pool_patch = patch("psycopg_pool.ConnectionPool", return_value=MagicMock())
_pool_patch.start()

