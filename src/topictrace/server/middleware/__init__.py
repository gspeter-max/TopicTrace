import time
import uuid
from collections import defaultdict

import structlog.contextvars
from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from topictrace import log
from topictrace.db.postgres.client import generate_key_hash, pool
from topictrace.server.app import app

# ---------------------------------------------------------------------------
# Request ID tracing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Generate or extract request ID, bind to structlog context, return in response."""
    incoming_id = request.headers.get("X-Request-ID")
    req_id = incoming_id if incoming_id else uuid.uuid4().hex[:12]

    structlog.contextvars.bind_contextvars(request_id=req_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id

    return response


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, 10 req/min per IP)
# ---------------------------------------------------------------------------

_rate_limit_dict = defaultdict(list)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_host = request.client.host
    now = time.time()

    _rate_limit_dict[client_host] = [
        rt for rt in _rate_limit_dict[client_host] if (now - rt) < 60
    ]
    if len(_rate_limit_dict[client_host]) >= 10:
        raise HTTPException(status_code=429, detail="rate limit exceeds")

    _rate_limit_dict[client_host].append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

_SKIP_AUTH_PATHS = ("/health/live", "/api-keys")


@app.middleware("http")
async def authenticate(request: Request, call_next):
    """Check Authorization header against api_keys table."""
    if request.url.path in _SKIP_AUTH_PATHS:
        return await call_next(request)

    auth_header = request.headers.get("authorization")
    if not auth_header:
        return JSONResponse(
            status_code=401, content={"detail": "Missing Authorization header"}
        )

    api_key = auth_header.removeprefix("Bearer ").strip()
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "Empty API key"})

    parts = api_key.split("_", 1)
    if len(parts) != 2:
        return JSONResponse(status_code=401, content={"detail": "Invalid key format"})

    key_prefix, key_part = parts
    key_hash = generate_key_hash(key_part)

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id FROM api_keys
                       WHERE key_prefix = %s AND key_hash = %s AND is_active = TRUE""",
                    (key_prefix, key_hash),
                )
                record = cur.fetchone()

        if record is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        return await call_next(request)

    except Exception as e:
        log.error(f"[AUTH] error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Authentication error"})
