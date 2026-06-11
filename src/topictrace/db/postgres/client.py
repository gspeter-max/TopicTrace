import hashlib

from psycopg_pool import ConnectionPool

from topictrace import log, settings

pool = ConnectionPool(
    settings.DATABASE_CONFIG.POSTGRES.POSTGRES_URI, min_size=4, max_size=10
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
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS research_cache (
                        id SERIAL PRIMARY KEY,
                        query_hash TEXT UNIQUE NOT NULL,
                        result JSONB NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_research_cache_lookup
                    ON research_cache (query_hash)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at
                    ON research_cache (expires_at)
                """)
        log.info("[DATABASE] tables created")
    except Exception as e:
        log.error(f"[DATABASE] init failed: {e}")
        raise
