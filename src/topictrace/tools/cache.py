"""Cache system for web_fetch with 20-minute TTL."""

import hashlib
import datetime
from psycopg.types.json import Json
from topictrace import log, settings
from topictrace.db.client import pool


def generate_fetch_cache_key(query: str, url: str) -> str:
    """Generate cache key from query + URL. Same URL with different queries = different cache."""
    combined = f"{query}:{url}".strip().lower()
    hashed_val = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"cache:tool:web_fetch:{hashed_val}"


def load_from_cache(cache_key: str) -> str | None:
    """Load cache from database. Returns None on miss or error."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT result FROM research_cache
                       WHERE query_hash = %s AND expires_at > NOW()""",
                    (cache_key,),
                )
                record = cur.fetchone()
                return record[0] if record else None
    except Exception as e:
        log.error(f"[CACHE] load failed: {e}")
        return None


def save_to_cache(cache_key: str, result: str, ttl: int = settings.CACHE_TTL_SECONDS) -> None:
    """Save cache to database. result stored as JSONB (wrapped in dict)."""
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=ttl)
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO research_cache (query_hash, result, expires_at)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (query_hash) DO UPDATE
                       SET result = EXCLUDED.result, expires_at = EXCLUDED.expires_at""",
                    (cache_key, Json({"content": result}), expires_at),
                )
        log.info(f"[CACHE] saved | key={cache_key}")
    except Exception as e:
        log.error(f"[CACHE] save failed: {e}")
        raise
