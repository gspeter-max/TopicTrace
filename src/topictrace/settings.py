"""Central configuration and constants for TopicTrace."""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM Provider (OpenGateway — mimo-v2.5-pro)
# ============================================================
LLM_BASE_URL = "https://opengateway.gitlawb.com/v1"
LLM_MODEL = "mimo-v2.5-pro"
LLM_API_KEY = os.getenv("LLM_API_KEY")

# ============================================================
# Tavily Search API
# ============================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============================================================
# Startup validation — fail fast with clear message
# ============================================================
_required = {"LLM_API_KEY": LLM_API_KEY, "TAVILY_API_KEY": TAVILY_API_KEY}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise SystemExit(
        f"Missing required env vars: {', '.join(_missing)}\n"
        f"Copy .env.example to .env and fill in your keys."
    )

# ============================================================
# Jina Reader API
# ============================================================
JINA_READER_BASE_URL = "https://r.jina.ai/"

# ============================================================
# Session Storage
# ============================================================
SESSIONS_DIR = "sessions"

# ============================================================
# Cache
# ============================================================
CACHE_TTL_SECONDS = 20 * 60  # 20 minutes

# ============================================================
# Summarization
# ============================================================
SUMMARIZE_MAX_INPUT_CHARS = 8000
SUMMARIZE_MAX_TOKENS = 1024
SUMMARIZE_TEMPERATURE = 0.7

# ============================================================
# Search
# ============================================================
SEARCH_MAX_RESULTS = 10
SEARCH_SNIPPET_MAX_CHARS = 300

# ============================================================
# Fetch
# ============================================================
FETCH_TIMEOUT_SECONDS = 30
