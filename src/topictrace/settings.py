"""Central configuration and constants for TopicTrace."""

import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class embedding_model_config( BaseModel ):
    JINA_API_KEY : str 
    JINA_EMBEDDING_MODEL : str
    JINA_BASE_URL : str
    MAX_CONCURRENCY : int
    JINA_EMBEDDING_TASK : str


# 1. Create a strict blueprint for what every LLM needs
class llm_config(BaseModel):
    LLM_BASE_URL: str
    LLM_MODEL: str
    LLM_API_KEY: str

# 2. Create the main settings class to hold your different AIs
class AppSettings(BaseModel):
    DEEPSEEK_AI: llm_config
    MISTRAL_AI: llm_config

class reranker_config(BaseModel):
    VOYAGE_API_KEY : str
    VOYAGE_RERANK_URL : str
    VOYAGE_RERANK_MODEL : str 

class neo4j_config(BaseModel):
    NEO4J_URI : str 
    NEO4J_USER : str
    NEO4J_PASSWORD : str 

class postgres_config(BaseModel):
    POSTGRES_URI : str 
    POSTGRES_PASSWORD : str 

class database_config(BaseModel):
    NEO4J : neo4j_config
    POSTGRES : postgres_config


EMBEDDING_CONFIG = embedding_model_config(
    JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip(),
    JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en",
    JINA_BASE_URL = "https://api.jina.ai/v1/embeddings",
    MAX_CONCURRENCY= 10,
    JINA_EMBEDDING_TASK = "retrieval.query"
)

DATABASE_CONFIG= database_config(
    NEO4J = neo4j_config(
        NEO4J_URI= os.getenv("NEO4J_URI"), 
        NEO4J_USER=os.getenv("NEO4J_USER"),
        NEO4J_PASSWORD= os.getenv("NEO4J_PASSWORD")
    ), 
    POSTGRES = postgres_config(
        POSTGRES_URI=os.getenv("POSTGRES_URI"),
        POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
    )
)

RERANKER_CONFIG= reranker_config(
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip(), 
    VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank", 
    VOYAGE_RERANK_MODEL = "rerank-2"
)

LLAMA_PARSE_APIKEY = os.getenv("LLAMA_PARSE_APIKEY", "").strip()

# 3. Fill the class with your specific data
LLM_CONFIG = AppSettings(
    DEEPSEEK_AI=llm_config(
        LLM_BASE_URL="https://integrate.api.nvidia.com/v1",
        LLM_MODEL="deepseek-ai/deepseek-v4-flash",
        LLM_API_KEY=os.getenv("LLM_API_KEY", "").strip()
    ),
    MISTRAL_AI=llm_config(
        LLM_BASE_URL="https://api.mistral.ai/v1",
        LLM_MODEL="mistral-small-latest",
        LLM_API_KEY=os.getenv("MISTRAL_API_KEY", "").strip()
    )
)


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

_required = {
    "LLM_API_KEY": LLM_CONFIG.MISTRAL_AI.LLM_API_KEY or LLM_CONFIG.DEEPSEEK_AI.LLM_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY, 
    "DATABASE_URL": DATABASE_URL,
    "LLAMA_PARSE_APIKEY" : LLAMA_PARSE_APIKEY,
    "JINA_API_KEY" : EMBEDDING_CONFIG.JINA_API_KEY,
    "VOYAGE_API_KEY" : RERANKER_CONFIG.VOYAGE_API_KEY,
    "NEO4J_USER" : DATABASE_CONFIG.NEO4J.NEO4J_USER,
    "NEO4J_URI" : DATABASE_CONFIG.NEO4J.NEO4J_URI,
    "NEO4J_PASSWORD" : DATABASE_CONFIG.NEO4J.NEO4J_PASSWORD
}

_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise SystemExit(
        f"Missing required env vars: {', '.join(_missing)}\n"
        f"Copy .env.example to .env and fill in your keys."
    )

JINA_READER_BASE_URL = "https://r.jina.ai/"
SESSIONS_DIR = "sessions"
CACHE_TTL_SECONDS = 20 * 60  # 20 minutes

SUMMARIZE_MAX_INPUT_CHARS = 8000
SUMMARIZE_MAX_TOKENS = 1024
SUMMARIZE_TEMPERATURE = 0.7

SEARCH_MAX_RESULTS = 10
SEARCH_SNIPPET_MAX_CHARS = 300
FETCH_TIMEOUT_SECONDS = 30

# RAG & Neo4j Settings
NEO4J_INDEX_NAME = "chunk_vector_index"
EMBEDDING_DIM = 768

# Entity Resolution Thresholds
ENTITY_RESOLUTION_FUZZY_THRESHOLD = 88
ENTITY_RESOLUTION_HIGH_THRESHOLD = 0.90
ENTITY_RESOLUTION_LOW_THRESHOLD = 0.50
ENTITY_RESOLUTION_DEFAULT_CANDIDATE_SCORE = 0.75

# Contextual Retrieval & LLM Generation Parameters
CONTEXTUAL_RETRIEVAL_MAX_TOKENS = 120
CONTEXTUAL_RETRIEVAL_MAX_CONCURRENCY = 10
LLM_CLIENT_TIMEOUT_SECONDS = 60


TOKENIZER_MODEL = "jinaai/jina-embeddings-v3"
CHUNK_SIZE= 512
CHUNK_OVERLAP = 100