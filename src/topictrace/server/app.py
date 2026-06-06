from fastapi import FastAPI
from contextlib import asynccontextmanager
from topictrace.server.routes.research import research_router
from topictrace.server.routes.api_key import router as api_key_router
from topictrace.db.client import init_db

from app.ingestionAPI import ingestionRouter
from app.retrieveAPI import retrieveRouter
 
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run init_db on startup, cleanup on shutdown."""
    init_db()
    print("[DATABASE] init_db is executed")
    yield


app = FastAPI(title="TopicTrace", lifespan=lifespan)
app.include_router(research_router)
app.include_router(api_key_router)
app.include_router(ingestionRouter)
app.include_router(retrieveRouter)

@app.get("/health/live")
async def health_check() -> dict:
    return {"status": "ok"}


# Import AFTER app is created — triggers @app.middleware registration
import topictrace.server.middleware  # noqa: E402

