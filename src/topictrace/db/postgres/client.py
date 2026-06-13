import hashlib

from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from topictrace import log, settings

connection_kwargs = {"autocommit": True, "row_factory": dict_row}
pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
    settings.DATABASE_CONFIG.POSTGRES.POSTGRES_URI,
    min_size=4,
    max_size=10,
    kwargs=connection_kwargs,
)


def generate_key_hash(key_part: str) -> str:
    """Hash an API key part with SHA-256 for secure storage/comparison."""
    return hashlib.sha256(key_part.encode()).hexdigest()


def init_postgres_db():
    """Create tables if they don't exist."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        key_prefix TEXT NOT NULL,
                        key_hash TEXT UNIQUE NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)  # type: ignore
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS research_cache (
                        id SERIAL PRIMARY KEY,
                        query_hash TEXT UNIQUE NOT NULL,
                        result JSONB NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)  # type: ignore
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS memory(
                        session_id TEXT,
                        memory_information TEXT
                )"""
                )
                cur.execute(
                    """ CREATE INDEX IF NOT EXISTS idx_memory  on memory( session_id ) """
                )
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_research_cache_lookup
                    ON research_cache (query_hash)
                """)  # type: ignore
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at
                    ON research_cache (expires_at)
                """)  # type: ignore
        log.info("[DATABASE] tables created")
    except Exception as e:
        log.error(f"[DATABASE] init failed: {e}")
        raise
