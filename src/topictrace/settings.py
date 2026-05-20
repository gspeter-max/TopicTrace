"""
Central configuration for TopicTrace.

All constants, API endpoints, and tunable values live here.
Import as: from topictrace import settings
Use as: settings.NVIDIA_MODEL
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# NVIDIA NIM API
# ============================================================
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "z-ai/glm-5.1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# ============================================================
# Tavily Search API
# ============================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
