import secrets

from fastapi import APIRouter

from topictrace.db.postgres.client import generate_key_hash, pool

router = APIRouter()


@router.post("/api-keys")
async def generate_api_key():
    random_part = secrets.token_urlsafe(32)
    key_prefix = "tt"
    full_key = f"{key_prefix}_{random_part}"
    key_hash = generate_key_hash(random_part)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO api_keys (key_prefix, key_hash, is_active)
                   VALUES (%s, %s, TRUE)""",
                (key_prefix, key_hash),
            )

    return {"key": full_key}
