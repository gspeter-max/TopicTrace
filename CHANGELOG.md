# Changelog

All notable changes to **TopicTrace** are documented in this file.

---

## [0.2.0] — 2026-06-07

### 🎯 Summary

Adds **Hybrid Adaptive RAG pipeline** — document ingestion into a **Neo4j knowledge graph**, vector + graph hybrid retrieval via a **LangGraph state machine** with intent routing, chunk grading, graph escalation, and Voyage AI reranking. Multi-LLM support (DeepSeek + Mistral), Jina embeddings, LlamaParse parsing, entity resolution with fuzzy matching. Original deep research agent preserved under `/research`.

---

### 🏗️ RAG Retrieval Architecture

```
POST /retrieve/query
  → route_query (LLM: "simple" or "complex")
  → vector_search (Jina → Neo4j cosine)
    ↓ complex              ↓ simple
    graph_search         grade_chunks
    ↓                      ↓ sufficient    ↓ not sufficient
    rerank (Voyage)      answer (fast)    graph_search → rerank → answer
    ↓
    answer → END
```

Simple queries skip graph + reranker entirely (fast path — saves ~2 LLM calls).

### Document Ingestion Pipeline

```
POST /ingestion/ingest
  → LlamaParse → chunk (512tok/100overlap) → contextual retrieval (LLM)
    → Jina embed (768d) → graph extraction (LLM: entities+rels)
      → entity resolution (exact→fuzzy→LLM) → Neo4j persist
```

---

### ✅ What Was Added

#### **RAG Ingestion (`rag/documentIngestion/`)**

| File | What it does |
|---|---|
| `parseDocument.py` | LlamaParse SDK — `parse(tier="cost_effective", expand=["text","items"])`, returns `{page, text, items}` per page |
| `chunking/recursive.py` | `RecursiveCharacterTextSplitter` (512tok, 100 overlap), Jina v3 tokenizer via `XLMRobertaTokenizerFast` (`@lru_cache`) |
| `contextual_retrieval.py` | LLM generates 1-3 retrieval sentences per chunk, produces `contextualized_text = context + chunk`. Semaphore-limited (10 concurrent) |
| `graphExtraction.py` | LLM extracts `{entity_name, entity_type, chunk_id, evidence_text}` + relationships. Constrained types: `RELATED_TO`, `PART_OF`, `DEPENDS_ON`, `ASSIGNED_TO`, `BLOCKED_BY`, `OWNS`, `REPORTS_TO`, `LOCATED_IN`. Invalid → `RELATED_TO` |
| `entityResolution.py` | Stage 1: exact normalization (`lower().strip().split()`, shortest=canonical). Stage 2: `rapidfuzz.token_set_ratio` ≥88 → ≥0.90 auto-merge, 0.50–0.90 LLM disambiguate, <0.50 reject |
| `graphPersistence.py` | `generate_stable_entity_id()` — SHA-256[:12]. Rewrites raw→canonical names. Builds payload: entities, mentions, relationships (deduped by `(src,type,tgt)`), `chunk_to_entity_ids` map |
| `graphRelationshipSchema.py` | Allowed relationship types list + prompt formatter |
| `models/graphExtractionModels.py` | Pydantic: `ExtractedEntity`, `ExtractedRelationship`, `ChunkGraphExtractionResult`, `CanonicalGraphPersistencePayload`, `EntityResolutionDecision` |
| `ingestion.py` | Orchestrator: parse → chunk → contextualize → embed → extract graph → resolve entities → persist. Returns `{raw_entity_count, canonical_entity_count, relationship_count}` |

#### **RAG Retrieval (`rag/documentRetrieve/`)**

| File | What it does |
|---|---|
| `router.py` | Mistral LLM classifies intent → `"simple"` / `"complex"` (JSON output, temp=0). Defaults `"simple"` on error |
| `grader.py` | Mistral evaluates chunk sufficiency → `GraderResult{sufficient, reason, answer}`. Fast-path: generates answer when sufficient=True (saves 1 LLM call). Defaults insufficient on error |
| `graphAgent.py` | 1-hop Neo4j traversal from entity IDs → human-readable fact sheet |
| `retrieve.py` | Entry point: creates Neo4j client, invokes LangGraph, maps to `QueryResponse`, closes in `finally`. Graph compiled once at module load |
| `graph/state.py` | `RAGState` TypedDict — 14 fields: query, top_k, intent, raw_chunks, vector_texts, grade_*, graph_facts, final_context, answer |
| `graph/nodes.py` | 6 nodes: `route_query`, `vector_search` (Jina→Neo4j), `grade_chunks_node`, `graph_search` (entity→1-hop), `rerank` (Voyage), `answer_node` (fast/standard path) |
| `graph/edges.py` | `route_after_vector_search`: complex→graph, simple→grade. `route_after_grader`: sufficient→answer, not→graph |
| `graph/build_graph.py` | Compiles `StateGraph(RAGState)` with 6 nodes + conditional edges |

#### **Neo4j Database (`db/neo4j/`)**

- `Neo4jClient` — wraps `AsyncGraphDatabase.driver`, async `execute_query()`, `close()`
- `cypherQuerys.py` — vector index creation (cosine, `IF NOT EXISTS`), chunk/document/entity upserts, batch entity+mention+relationship writes, `retrieve_similar_chunks()` (vector search), `fetch_entity_neighbors_1hop()`

#### **Postgres Restructure (`db/postgres/`)**

Moved `db/client.py` → `db/postgres/client.py`. Renamed `init_db()` → `init_postgres_db()`. Same functionality.

#### **Multi-LLM Providers (`provider/`)**

- `llm.py` — `get_llm(provider)` accepts `"DEEPSEEK_AI"` | `"MISTRAL_AI"`. DeepSeek enables thinking mode; Mistral does not
- `embedding.py` — NEW — `embeddingModel` class wrapping Jina Embeddings API (`jina-embeddings-v2-base-en`, 768d). Async semaphore-limited
- `rerank.py` — NEW — `rerank_documents()` via Voyage AI (`rerank-2`). Async httpx, sorts by `relevance_score`, returns top-k

#### **Prompts (`prompts/`)**

- `research_agent.py` — renamed from `agent_system.py` (same content)
- `router_intent_classifier.py` — NEW — classifies query → `"simple"` / `"complex"` (JSON)
- `grader_chunk_evaluator.py` — NEW — evaluates sufficiency + generates fast-path answer (JSON)
- `answer_generator.py` — NEW — final answer from reranked context only
- `ingestion/extraction.py` — NEW — entity+relationship extraction (one-shot JSON example, enforces relationship schema)
- `ingestion/resolution.py` — NEW — entity disambiguation (merge/reject decisions, JSON)

#### **Settings Overhaul (`settings.py`)**

Refactored to Pydantic `BaseModel` configs: `llm_config`, `AppSettings` (DEEPSEEK_AI + MISTRAL_AI), `embedding_model_config` (Jina), `reranker_config` (Voyage), `neo4j_config`, `postgres_config`, `database_config`.

New constants: `NEO4J_INDEX_NAME="chunk_vector_index"`, `EMBEDDING_DIM=768`, entity resolution thresholds (fuzzy=88, high=0.90, low=0.50, default=0.75), `CONTEXTUAL_RETRIEVAL_MAX_TOKENS=120`, `CONTEXTUAL_RETRIEVAL_MAX_CONCURRENCY=10`, `LLM_CLIENT_TIMEOUT_SECONDS=60`.

New required env vars: `LLAMA_PARSE_APIKEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, `MISTRAL_API_KEY`, `NEO4J_USER`, `NEO4J_URI`, `NEO4J_PASSWORD`.

#### **API Routes Restructure (`server/`)**

- `routes/deep_research/research.py` — original `POST /research` + `POST /research/stream` (SSE) moved here
- `routes/rag/ingestionAPI.py` — NEW — `POST /ingestion/ingest` → `{status, message, raw_entity_count, canonical_entity_count, relationship_count}`
- `routes/rag/retrieveAPI.py` — NEW — `POST /retrieve/query` → `{answer, intent, used_graph_search, reason_for_graph_search, context_used}`
- Schemas split: `schemas/deep_research/research.py`, `schemas/rag/ingestionModels.py`, `schemas/rag/retrieveModels.py`
- `app.py` — lifespan inits Postgres + Neo4j (vector index), registers 4 routers

#### **Docker**

- `docker-compose.yml` — added Neo4j 5.26 Community (ports 7474/7687, env auth, 512MB/1GB heap, vector index enabled, persistent volumes, health check)
- `.env.example` — added Neo4j, LlamaParse, Jina, Voyage, Mistral vars

#### **New Dependencies**

| Package | Version | Purpose |
|---|---|---|
| `llama-cloud` | ≥2.0.0 | LlamaParse document parsing |
| `transformers` | ≥4.40.0 | Jina tokenizer for token counting |
| `langchain-text-splitters` | ≥0.3.0 | Recursive text chunking |
| `rapidfuzz` | ≥3.9.0 | Fuzzy entity resolution |
| `neo4j` | ≥5.0.0 | Async Neo4j driver |
| `respx` | ≥0.21.0 | HTTP mocking for tests |
| `structlog` | ≥23.3.0 | Made explicit dependency |

#### **Testing**

- `tests/rag/documentIngestion/` — 7 test files + 2 subdirs: parsing, graph extraction, entity resolution, persistence, ID generation, pipeline integration, prompt contracts, chunking, contextual retrieval
- `tests/rag/retrieve/` — 10 test files: router, grader (+ deep), graph agent, RAG graph, retrieve pipeline (+ deep), Voyage reranker (+ deep), config
- `tests/rag/real_env_test/` — real environment integration: ingestion + retrieval with live APIs
- `pyproject.toml` — added `integration` marker, starlette deprecation filter

#### **Other**

- `data/pankajkumar.pdf` — sample ingestion doc
- `graphify-out/` — code analysis output (GRAPH_REPORT.md, graph.html/json)

---

### 📁 File Tree

```
src/
└── topictrace/
    ├── __init__.py                      # structlog setup
    ├── settings.py                      # Pydantic config (multi-LLM, Neo4j, Jina, Voyage)
    │
    ├── db/
    │   ├── __init__.py
    │   ├── neo4j/
    │   │   ├── __init__.py              # Neo4jClient (AsyncGraphDatabase)
    │   │   └── cypherQuerys.py          # Cypher: vector index, CRUD, similarity search, 1-hop
    │   └── postgres/
    │       └── client.py                # Pool, tables, key hashing
    │
    ├── provider/
    │   ├── __init__.py
    │   ├── llm.py                       # ChatOpenAI factory (DeepSeek + Mistral)
    │   ├── embedding.py                 # Jina Embeddings v2 (768d)
    │   └── rerank.py                    # Voyage AI reranker (rerank-2)
    │
    ├── agents/
    │   ├── state.py                     # ResearchState TypedDict
    │   └── graph.py                     # Deep research LangGraph (AGENT ↔ TOOLS)
    │
    ├── prompts/
    │   ├── __init__.py
    │   ├── research_agent.py            # System + user prompt (3 depth levels)
    │   ├── router_intent_classifier.py  # simple/complex classification
    │   ├── grader_chunk_evaluator.py    # Chunk sufficiency + fast-path answer
    │   ├── answer_generator.py          # Final answer from context
    │   └── ingestion/
    │       ├── extraction.py            # Entity + relationship extraction
    │       └── resolution.py            # Entity disambiguation
    │
    ├── tools/
    │   ├── __init__.py
    │   ├── cache.py                     # Postgres cache (SHA-256, 20-min TTL)
    │   ├── web_search.py                # Tavily (async, batch, @tool)
    │   └── web_fetch.py                 # Jina fetch + LLM summarize + cache
    │
    ├── rag/
    │   ├── documentIngestion/
    │   │   ├── __init__.py
    │   │   ├── ingestion.py             # 6-stage pipeline orchestrator
    │   │   ├── parseDocument.py         # LlamaParse → pages
    │   │   ├── contextual_retrieval.py  # LLM retrieval context per chunk
    │   │   ├── graphExtraction.py       # LLM entity+relationship extraction
    │   │   ├── entityResolution.py      # exact → fuzzy → LLM disambiguation
    │   │   ├── graphPersistence.py      # Canonical rewriting + Neo4j payload
    │   │   ├── graphRelationshipSchema.py
    │   │   ├── chunking/
    │   │   │   ├── __init__.py
    │   │   │   └── recursive.py         # RecursiveCharacterTextSplitter
    │   │   └── models/
    │   │       └── graphExtractionModels.py  # Pydantic models
    │   │
    │   └── documentRetrieve/
    │       ├── retrieve.py              # Entry point: handle_query()
    │       ├── router.py                # Intent classifier
    │       ├── grader.py                # Chunk sufficiency evaluator
    │       ├── graphAgent.py            # 1-hop graph traversal
    │       └── graph/
    │           ├── __init__.py
    │           ├── build_graph.py       # LangGraph compiler
    │           ├── state.py             # RAGState (14 fields)
    │           ├── nodes.py             # 6 nodes
    │           └── edges.py             # 2 conditional edges
    │
    └── server/
        ├── __init__.py
        ├── app.py                       # FastAPI (Postgres+Neo4j init, 4 routers)
        ├── middleware/
        │   └── __init__.py              # Request ID, rate limit, CORS, API key auth
        ├── routes/
        │   ├── __init__.py
        │   ├── api_key.py               # POST /api-keys
        │   ├── deep_research/
        │   │   └── research.py          # POST /research + /research/stream
        │   └── rag/
        │       ├── ingestionAPI.py       # POST /ingestion/ingest
        │       └── retrieveAPI.py        # POST /retrieve/query
        └── schemas/
            ├── deep_research/
            │   └── research.py
            └── rag/
                ├── ingestionModels.py
                └── retrieveModels.py

tests/
├── __init__.py
├── conftest.py
├── test_project_structure.py
├── test_deep_audit.py
├── tools/
│   └── __init__.py
└── rag/
    ├── __init__.py
    ├── documentIngestion/
    │   ├── __init__.py
    │   ├── test_parse_document.py
    │   ├── test_graph_extraction.py
    │   ├── test_entity_resolution.py
    │   ├── test_graph_persistence.py
    │   ├── test_id_generation.py
    │   ├── test_ingestion_pipeline.py
    │   ├── test_prompt_contracts.py
    │   ├── chunking/
    │   └── contextual_retrieval/
    ├── retrieve/
    │   ├── __init__.py
    │   ├── test_config.py
    │   ├── test_router.py
    │   ├── test_grader.py
    │   ├── test_grader_deep.py
    │   ├── test_graph_agent.py
    │   ├── test_rag_graph.py
    │   ├── test_retrieve_pipeline.py
    │   ├── test_retrieve_pipeline_deep.py
    │   ├── test_voyage_reranker.py
    │   └── test_voyage_reranker_deep.py
    └── real_env_test/
        ├── __init__.py
        ├── run_real_env_test.py
        ├── run_retrieve_real_env_test.py
        └── test_run_real_env_test.py
```

---
---

## [0.1.0] — 2026-06-06

### 🎯 Summary

Initial release — AI-powered research agent for exam prep. Accepts questions via REST API, searches web (Tavily), fetches+summarizes pages (Jina Reader), caches in PostgreSQL, returns structured answers. LangGraph state machine powered by DeepSeek V4 Flash via NVIDIA API.

---

### 🏗️ Core Architecture

```
Client → FastAPI → Middleware (reqID → rate limit → CORS → auth)
  → Research Endpoint → LangGraph (AGENT ↔ TOOLS)
    → LLM needs data? → web_search → web_fetch → summarize → cache → loop
    → LLM done? → return answer
```

---

### ✅ What Was Added

#### **Logging (`__init__.py`)**
structlog with timestamp (HH:MM:SS), log level, callsite (file/func/line), context vars (`request_id`), colored console. Single `log` object used everywhere.

#### **Config (`settings.py`)**
Loads `.env` via `load_dotenv()`. Required: `LLM_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`. Fails fast on missing vars. Constants: `LLM_BASE_URL` (NVIDIA), `LLM_MODEL` (deepseek-v4-flash), `JINA_READER_BASE_URL`, `CACHE_TTL_SECONDS` (20min), `SUMMARIZE_MAX_INPUT_CHARS` (8000), `SUMMARIZE_MAX_TOKENS` (1024), `SUMMARIZE_TEMPERATURE` (0.7), `SEARCH_MAX_RESULTS` (10), `SEARCH_SNIPPET_MAX_CHARS` (300), `FETCH_TIMEOUT_SECONDS` (30).

#### **Database (`db/client.py`)**
`psycopg_pool.ConnectionPool` (min=4, max=10). `generate_key_hash()` — SHA-256 API key hashing. `init_db()` creates `api_keys` (id, key_prefix, key_hash UNIQUE, is_active, created_at) and `research_cache` (id, query_hash UNIQUE, result JSONB, expires_at, created_at) with indexes on `query_hash` and `expires_at`.

#### **LLM Provider (`provider/llm.py`)**
`get_llm()` — `ChatOpenAI` pointing at NVIDIA/DeepSeek. Custom httpx clients with `Accept-Encoding: identity` (workaround for broken gzip), 60s timeout, thinking mode enabled (`reasoning_effort: "high"`).
`get_llm_with_tools(tools)` — binds tool schemas for LangGraph `ToolNode`.

#### **Agent (`agents/`)**
- `state.py` — `ResearchState` TypedDict with `messages` annotated `add_messages` (append, not replace)
- `graph.py` — LangGraph: `AGENT` node (system prompt + LLM) ↔ `TOOLS` node (ToolNode). `should_continue`: tool_calls → TOOLS, else → END. Wiring: START→AGENT→(conditional)→TOOLS/END, TOOLS→AGENT

#### **Prompts (`prompts/agent_system.py`)**
`get_system_prompt()` — identity, tool usage order, good/bad query examples, output rules (cite sources, Markdown, exam-relevant). Never answer from memory for specific papers.
`get_user_prompt(query, depth)` — three profiles: quick (1 search/1 fetch/bullets), standard (2/3/structured), deep (4/6/comprehensive multi-angle).

#### **Tools**
- `web_search.py` — Tavily @tool. Batch via `asyncio.gather`, input validation, API key guard, `{title, url, snippet}` (300 char), error isolation per query
- `web_fetch.py` — Jina Reader @tool. Cache-first (SHA-256 key, Postgres, 20-min TTL), parallel fetch, LLM summarization (8000 char truncate), cache write-back, error handling (403/404/451)
- `cache.py` — `generate_fetch_cache_key(query, url)` (SHA-256, prefix `cache:tool:web_fetch:`), `load_from_cache()` (TTL-aware, fail-open), `save_to_cache()` (upsert, JSONB)

#### **Server (`server/`)**
- `app.py` — lifespan runs `init_db()`, mounts routers, `GET /health/live`, middleware import after app creation
- `middleware/` — request ID (12-char hex), rate limit (10/min/IP sliding window), CORS (*), API key auth (SHA-256 hash lookup, skips `/health/live` + `/api-keys`)
- `POST /research` — `{query, depth}` → LangGraph → `{answer}`
- `POST /research/stream` — SSE via `astream_events(v2)`. Tokens: `{token}`, tool calls: `{status, tool}`, errors: `{status, error}`, end: `[DONE]`
- `POST /api-keys` — `secrets.token_urlsafe(32)`, format `tt_<secret>`, stores hash only

#### **Schemas (`schemas/research.py`)**
`ResearchRequest{query, depth="quick"}`, `ResearchResponse{answer}`. Pydantic validation, 422 on invalid.

#### **Docker**
- `Dockerfile` — python:3.11, curl, uv sync, uvicorn:8080 --reload, HEALTHCHECK
- `docker-compose.yml` — postgres:16 (pgdata volume, pg_isready), topictrace (src mount, depends_on healthy)
- `.env.example` — LLM_API_KEY, TAVILY_API_KEY

#### **Testing**
`conftest.py` (fixtures), `test_project_structure.py`, `test_deep_audit.py` (81 tests), `tools/` tests.

#### **Dependencies**
tavily-python ≥0.5.0, structlog, requests ≥2.31.0, openai ≥1.50.0, python-dotenv ≥1.0.0, rich ≥13.0.0, prompt-toolkit ≥3.0.0, httpx ≥0.27.0, langgraph ≥1.2.2, langchain-core ≥1.4.0, langchain-openai ≥1.2.2, fastapi ≥0.136.3, uvicorn ≥0.48.0, psycopg[binary] ≥3.3.4, psycopg-pool ≥3.3.1. Dev: pytest ≥8.0.0, pytest-cov ≥5.0.0. Build: Hatchling.
