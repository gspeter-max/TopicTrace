from fastapi import FastAPI
from contextlib import asynccontextmanager
from topictrace import settings, log 
from topictrace.db.neo4j import Neo4jClient
from topictrace.server.routes.deep_research.research import research_router
from topictrace.server.routes.api_key import router as api_key_router
from topictrace.db.postgres.client import init_postgres_db
from topictrace.db.neo4j.cypherQuerys import create_vector_index

from topictrace.server.routes.rag.ingestionAPI import ingestionRouter
from topictrace.server.routes.rag.retrieveAPI import retrieveRouter

 
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize databases on startup, clean up on shutdown."""
    neo4j_client: Neo4jClient | None = None
    try:
        init_postgres_db()
        print("[DATABASE][POSTGRES] database is created")

        neo4j_client = Neo4jClient(
            settings.DATABASE_CONFIG.NEO4J.NEO4J_URI,
            settings.DATABASE_CONFIG.NEO4J.NEO4J_USER,
            settings.DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD,
        )
        await create_vector_index(neo4j_client, settings.NEO4J_INDEX_NAME, settings.EMBEDDING_DIM)
        print("[DATABASE][NEO4J] database is created")
    except Exception as e:
        log.warning("[DATABASE] database fail to initilize", error=str(e))

    yield  # app runs here; code after yield is shutdown

    if neo4j_client is not None:
        await neo4j_client.close()


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

