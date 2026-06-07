# Changelog

All notable changes to **TopicTrace** are documented in this file.

---

## [0.2.0] — 2026-06-07

### 🎯 Summary

Major release introducing the **Hybrid Adaptive RAG (Retrieval-Augmented Generation) pipeline**. TopicTrace now ingests documents into a **Neo4j knowledge graph**, performs **vector + graph hybrid retrieval**, and answers queries using a **multi-stage LangGraph state machine** with automatic intent routing, chunk grading, graph escalation, and Voyage AI reranking. The system now supports **two LLM providers** (DeepSeek AI via NVIDIA + Mistral AI), **Jina embeddings**, **LlamaParse document parsing**, and **entity resolution with fuzzy matching**. The original deep research agent (web search + fetch + summarize) is preserved under a dedicated `/research` route.

---

### 🏗️ New Architecture — Hybrid Adaptive RAG Pipeline

The retrieval pipeline implements an adaptive decision graph:

```
User Query
  → POST /retrieve/query
    → LangGraph State Machine (build_graph.py)
      → route_query (LLM classifies intent: "simple" or "complex")
      → vector_search (Jina embedding → Neo4j cosine similarity)
        ↓ (complex)                    ↓ (simple)
        graph_search               grade_chunks (LLM evaluates sufficiency)
        ↓                            ↓ (sufficient)    ↓ (not sufficient)
        rerank (Voyage AI)         answer_node        graph_search
        ↓                          (fast path)         ↓
        answer_node                                   rerank (Voyage AI)
        ↓                                             ↓
        END                                          answer_node → END
```

**Key design decision:** Simple queries that can be answered from vector chunks alone skip the graph and reranker entirely (fast path), saving ~2 LLM calls and ~1 API round-trip per request.

---

### Document Ingestion Pipeline

The ingestion pipeline converts a raw document into a searchable knowledge graph:

```
POST /ingestion/ingest {file_path: "..."}
  → LlamaParse (document → structured pages)
    → Recursive text chunking (512 tokens, 100 overlap)
      → Contextual retrieval (LLM generates retrieval context per chunk)
        → Jina embedding (chunk → 768-dim vector)
        → Graph extraction (LLM extracts entities + relationships per chunk)
          → Entity resolution (exact normalization → fuzzy merge → LLM disambiguation)
            → Graph persistence (Neo4j: Document → Chunk → Entity → RELATES_TO)
```

---

### ✅ What Was Added

#### **Hybrid Adaptive RAG Module (`src/topictrace/rag/`)**

Entirely new module implementing document ingestion and retrieval.

---

##### **Document Ingestion (`src/topictrace/rag/documentIngestion/`)**

**`parseDocument.py`** — Document parsing via LlamaParse (LlamaCloud SDK ≥ 2.0):
- Uses `LlamaCloud.parsing.parse()` with `tier="cost_effective"` and `expand=["text", "items"]`.
- Returns a normalized list of `{page, text, items}` dicts — one per page.
- `get_all_pages_text()` concatenates all pages with `--- Page N ---` markers for downstream processing.

**`chunking/recursive.py`** — Paragraph-aware, token-counted text chunking:
- Uses `RecursiveCharacterTextSplitter` from `langchain-text-splitters` with separators `["\n\n", "\n", ". ", " ", ""]`.
- Token counting via `XLMRobertaTokenizerFast` from `jinaai/jina-embeddings-v3` (cached with `@lru_cache`).
- Default: 512 tokens per chunk, 100 token overlap.
- Each chunk dict contains: `chunk_id` (format: `{document_id}::{index}`), `chunk_index`, `text`, `token_count`, `document_id`.

**`contextual_retrieval.py`** — LLM-generated retrieval context for each chunk:
- For each chunk, sends the full document text + chunk text to the LLM with a prompt: *"Write 1 to 3 short sentences that help retrieval. Mention the section, topic, or document role of this chunk."*
- Produces a `contextualized_text` field: `{context}\n\n{chunk_text}` — this is what gets embedded and stored.
- Concurrency-limited via `asyncio.Semaphore` (default: 10 concurrent LLM calls).

**`graphExtraction.py`** — LLM-based entity and relationship extraction:
- Sends each chunk to the LLM with a structured JSON output prompt requesting `entities` and `relationships` arrays.
- Entity schema: `{entity_name, entity_type, chunk_id, evidence_text}`.
- Relationship schema: `{source_entity_name, relationship_type, target_entity_name, chunk_id, evidence_text}`.
- Relationship types are constrained to an allowed list: `RELATED_TO`, `PART_OF`, `DEPENDS_ON`, `ASSIGNED_TO`, `BLOCKED_BY`, `OWNS`, `REPORTS_TO`, `LOCATED_IN`. Invalid types fall back to `RELATED_TO`.
- Uses Pydantic validation on `evidence_text` (cannot be empty) and `relationship_type` (must be from allowed list).

**`entityResolution.py`** — Two-stage entity deduplication:
- **Stage 1 — Exact normalization:** Groups names identical after `lower().strip().split()`. Picks the shortest variant as canonical. E.g., `["Apple Inc", "apple inc"]` → canonical `"Apple Inc"`.
- **Stage 2 — Fuzzy merge:** Uses `rapidfuzz.fuzz.token_set_ratio` with threshold 88 to find near-duplicate pairs. Pairs are split into three buckets:
  - `≥ 0.90` → auto-merge (clear same entity)
  - `0.50–0.90` → send to LLM for disambiguation
  - `< 0.50` → auto-reject (clearly different)
- LLM disambiguation uses a JSON prompt returning `{left_name, right_name, should_merge, canonical_name}` per pair.

**`graphPersistence.py`** — Canonical entity rewriting and Neo4j write payload construction:
- `generate_stable_entity_id(name)` — SHA-256 hash of normalized name, truncated to 12 hex characters. Deterministic: same name always produces the same ID.
- `rewrite_graph_results_to_canonical_entities()` — replaces all raw entity names with their resolved canonical names across entities and relationships.
- `build_neo4j_graph_write_payload()` — assembles the final write payload with:
  - Deduplicated `entities` list with `{canonical_name, entity_type, entity_id}`.
  - `mentions` list linking entities to chunks with evidence text.
  - Merged `relationships` list (deduplicated by `(source, type, target)` key).
  - `chunk_to_entity_ids` mapping — for each chunk, the list of entity IDs found in it (used for fast graph lookups during retrieval).

**`graphRelationshipSchema.py`** — Defines the `ALLOWED_RELATIONSHIP_TYPES` list and a helper to format it as prompt text.

**`models/graphExtractionModels.py`** — Pydantic models:
- `ExtractedEntity`, `ExtractedRelationship` — raw LLM output with validation.
- `ChunkGraphExtractionResult` — wrapper for entities + relationships per chunk.
- `CanonicalGraphPersistencePayload` — post-resolution entities + relationships.
- `EntityResolutionDecision` — LLM disambiguation result `{left_name, right_name, should_merge, canonical_name}`.

**`ingestion.py`** — Pipeline orchestrator:
- `ingest_document_graph(file_path, provider)` runs the full 6-stage pipeline: parse → chunk → contextualize → embed → extract graph → resolve entities → persist to Neo4j.
- Returns a summary response with `raw_entity_count`, `canonical_entity_count`, `relationship_count`.

---

##### **Document Retrieval (`src/topictrace/rag/documentRetrieve/`)**

**`router.py`** — LLM-based query intent classifier:
- Uses Mistral AI with `response_format={"type": "json_object"}` and `temperature=0.0`.
- Classifies queries as `"simple"` (single fact/definition lookup) or `"complex"` (relationship traversal, comparisons).
- Defaults to `"simple"` on any error — fail-safe.

**`grader.py`** — LLM chunk sufficiency evaluator:
- Evaluates whether retrieved vector chunks contain enough information to answer the query.
- Returns a `GraderResult` Pydantic model: `{sufficient: bool, reason: str, answer: str}`.
- **Fast-path optimization:** When `sufficient=True`, the grader also generates the final answer in the same LLM call, saving a second LLM round-trip.
- Defaults to `sufficient=False` on any error — errs on the side of safety (triggers graph escalation).

**`graphAgent.py`** — Neo4j knowledge graph traversal:
- `gather_graph_facts(client, entity_ids)` — performs 1-hop traversal from seed entity IDs.
- Formats results as a human-readable fact sheet: `"- Source RELATES_TO Target (Evidence: '...')"`.
- Returns empty string if no entities or no relationships found.

**`retrieve.py`** — Pipeline entry point:
- `handle_query(request)` creates a Neo4j client, invokes the LangGraph, maps final state to `QueryResponse`, and closes the client in a `finally` block.
- The graph is compiled once at module load (`_rag_graph = build_rag_graph()`) — it is stateless and reusable.

**`graph/state.py`** — `RAGState` TypedDict with all pipeline fields: `query`, `top_k`, `top_k_rerank`, `intent`, `raw_chunks`, `vector_texts`, `grade_sufficient`, `grade_reason`, `grade_answer`, `graph_facts`, `used_graph_search`, `reason_for_graph_search`, `final_context`, `answer`.

**`graph/nodes.py`** — Six LangGraph node functions:
1. `route_query` — classifies intent via Mistral.
2. `vector_search` — embeds query via Jina, runs cosine similarity search in Neo4j, returns `raw_chunks` + `vector_texts`.
3. `grade_chunks_node` — evaluates chunk sufficiency via Mistral.
4. `graph_search` — extracts entity IDs from chunks, traverses Neo4j graph for 1-hop relationships.
5. `rerank` — combines vector texts + graph facts, reranks via Voyage AI, returns `final_context`.
6. `answer_node` — fast path (reuse grader answer) or standard path (generate from reranked context via Mistral).

**`graph/edges.py`** — Two conditional edge functions:
- `route_after_vector_search` — complex → `graph_search`, simple → `grade_chunks`.
- `route_after_grader` — sufficient → `answer_node` (fast path), not sufficient → `graph_search` (escalation).

**`graph/build_graph.py`** — Compiles the `StateGraph(RAGState)` with all 6 nodes and wiring.

---

#### **Neo4j Database Layer (`src/topictrace/db/neo4j/`)**

**`__init__.py`** — `Neo4jClient` class:
- Wraps `AsyncGraphDatabase.driver` with async session management.
- `execute_query(query, parameters)` — runs a Cypher query in an async session and returns `list[dict]`.
- `close()` — cleanly shuts down the driver.

**`cypherQuerys.py`** — All Cypher queries for the knowledge graph:
- **Index:** `create_vector_index()` — creates a cosine-similarity vector index on `Chunk.embedding` with `IF NOT EXISTS` guard.
- **Write:**
  - `save_chunk()` — upserts a `:Chunk` node with text, context, embedding, and `entity_ids` list.
  - `save_document_node()` — upserts a `:Document` node.
  - `save_entity_nodes_and_relationships()` — batch writes `:Entity` nodes, `:MENTIONED_IN` edges, and `:RELATES_TO` edges.
- **Read:**
  - `retrieve_similar_chunks()` — vector similarity search via `db.index.vector.queryNodes`, returns `full_context`, `score`, `chunk_id`, `document_id`, `entity_ids`.
  - `fetch_entity_neighbors_1hop()` — 1-hop graph traversal from a list of entity IDs, returns `source`, `rel_type`, `target`, `evidence_text`.

---

#### **Postgres Database Restructure (`src/topictrace/db/postgres/`)**

- Moved from `db/client.py` → `db/postgres/client.py`.
- Renamed `init_db()` → `init_postgres_db()` to disambiguate from Neo4j init.
- Core functionality unchanged: connection pool, table creation, key hashing.

---

#### **Multi-LLM Provider Support (`src/topictrace/provider/`)**

**`llm.py`** — Refactored to support multiple providers:
- `get_llm(provider)` now accepts `Literal["DEEPSEEK_AI", "MISTRAL_AI"]` parameter (default: `"DEEPSEEK_AI"`).
- Reads provider-specific config from `settings.LLM_CONFIG` using `getattr(settings.LLM_CONFIG, provider)`.
- DeepSeek AI enables thinking mode (`reasoning_effort: "high"`); Mistral AI does not.
- Both share the same `Accept-Encoding: identity` header workaround.

**`embedding.py`** — NEW — Jina embedding model client:
- `embeddingModel` class wraps the Jina Embeddings API (`jina-embeddings-v2-base-en`).
- `generateEmebedding(texts)` — accepts a string or list of strings, returns `list[list[float]]`.
- Concurrency-limited via `asyncio.Semaphore`.
- Uses `requests.post` (synchronous) under an async semaphore for API calls.

**`rerank.py`** — NEW — Voyage AI reranker:
- `rerank_documents(query, documents, top_k)` — async function calling `https://api.voyageai.com/v1/rerank` with `rerank-2` model.
- Sorts results by `relevance_score` descending, returns top-k document strings.
- Uses `httpx.AsyncClient` for native async HTTP.

---

#### **Extended Prompt Engineering (`src/topictrace/prompts/`)**

**Restructured:** The old `agent_system.py` prompt was renamed and moved to `research_agent.py`. Content is functionally identical.

**New prompt modules:**

**`router_intent_classifier.py`** — System prompt for the query intent router:
- Classifies queries as `"simple"` (facts, definitions, summaries) or `"complex"` (connections, relationships, hierarchies).
- Requires JSON output: `{"intent": "simple"}`.

**`grader_chunk_evaluator.py`** — System prompt for the chunk sufficiency grader:
- Dual-purpose: evaluates sufficiency AND generates the answer on the fast path.
- Requires JSON output: `{"sufficient": bool, "reason": str, "answer": str}`.

**`answer_generator.py`** — System prompt for final answer generation:
- `build_final_answer_prompt(context_block)` — injects reranked context into a strict "answer from context only" prompt.

**`ingestion/extraction.py`** — System + user prompts for knowledge graph extraction:
- Instructs the LLM to extract entities and relationships from chunk text.
- Includes a one-shot JSON example for output format.
- Enforces the allowed relationship type schema.

**`ingestion/resolution.py`** — System + user prompts for entity disambiguation:
- Asks the LLM to decide if ambiguous name pairs refer to the same entity.
- Requires JSON output: `{"decisions": [{left_name, right_name, should_merge, canonical_name}]}`.

---

#### **Centralized Configuration Overhaul (`src/topictrace/settings.py`)**

Major refactor using **Pydantic `BaseModel`** for type-safe, validated configuration:

- **`llm_config`** — blueprint for any LLM: `{LLM_BASE_URL, LLM_MODEL, LLM_API_KEY}`.
- **`AppSettings`** — holds named LLM configs: `DEEPSEEK_AI` and `MISTRAL_AI`.
- **`embedding_model_config`** — Jina embedding settings: `{JINA_API_KEY, JINA_EMBEDDING_MODEL, JINA_BASE_URL, MAX_CONCURRENCY, JINA_EMBEDDING_TASK}`.
- **`reranker_config`** — Voyage AI settings: `{VOYAGE_API_KEY, VOYAGE_RERANK_URL, VOYAGE_RERANK_MODEL}`.
- **`neo4j_config`** — Neo4j connection: `{NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD}`.
- **`postgres_config`** — Postgres connection: `{POSTGRES_URI, POSTGRES_PASSWORD}`.
- **`database_config`** — wraps both `NEO4J` and `POSTGRES` configs.

**New settings constants:**
| Constant | Value | Purpose |
|---|---|---|
| `NEO4J_INDEX_NAME` | `"chunk_vector_index"` | Name of the Neo4j vector index |
| `EMBEDDING_DIM` | `768` | Jina embedding dimensionality |
| `ENTITY_RESOLUTION_FUZZY_THRESHOLD` | `88` | Minimum fuzzy score to consider a merge candidate |
| `ENTITY_RESOLUTION_HIGH_THRESHOLD` | `0.90` | Auto-merge threshold |
| `ENTITY_RESOLUTION_LOW_THRESHOLD` | `0.50` | Auto-reject threshold |
| `ENTITY_RESOLUTION_DEFAULT_CANDIDATE_SCORE` | `0.75` | Default score for fuzzy candidates |
| `CONTEXTUAL_RETRIEVAL_MAX_TOKENS` | `120` | Max tokens for contextual retrieval LLM calls |
| `CONTEXTUAL_RETRIEVAL_MAX_CONCURRENCY` | `10` | Max parallel LLM calls for contextual retrieval |
| `LLM_CLIENT_TIMEOUT_SECONDS` | `60` | HTTP client timeout for LLM calls |

**New required env vars:** `LLAMA_PARSE_APIKEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, `MISTRAL_API_KEY`, `NEO4J_USER`, `NEO4J_URI`, `NEO4J_PASSWORD`.

---

#### **API Routes Restructure (`src/topictrace/server/`)**

Routes and schemas reorganized into feature-based subdirectories:

**`routes/deep_research/research.py`** — Original research endpoints moved here:
- `POST /research` — synchronous deep research.
- `POST /research/stream` — SSE streaming deep research.

**`routes/rag/ingestionAPI.py`** — NEW:
- `POST /ingestion/ingest` — accepts `{file_path: str}`, runs the full document-to-knowledge-graph pipeline, returns `{status, message, raw_entity_count, canonical_entity_count, relationship_count}`.

**`routes/rag/retrieveAPI.py`** — NEW:
- `POST /retrieve/query` — accepts `{query: str, top_k: int, top_k_rerank: int}`, runs the Hybrid Adaptive RAG pipeline, returns `{answer, intent, used_graph_search, reason_for_graph_search, context_used}`.

**`schemas/deep_research/research.py`** — Research request/response models (moved from `schemas/research.py`).

**`schemas/rag/ingestionModels.py`** — NEW:
- `IngestionRequest` — `{file_path: str}`.
- `IngestionResponse` — `{status, message, raw_entity_count, canonical_entity_count, relationship_count}`.

**`schemas/rag/retrieveModels.py`** — NEW:
- `QueryRequest` — `{query: str, top_k: int = 5, top_k_rerank: int = 3}`.
- `QueryResponse` — `{answer, intent, used_graph_search, reason_for_graph_search, context_used: list[str]}`.

**`app.py`** — Updated lifespan to initialize both Postgres AND Neo4j (vector index creation), registers new routers: `ingestionRouter`, `retrieveRouter`.

---

#### **Docker & Deployment**

**`docker-compose.yml`** — Added **Neo4j 5.26 Community** service:
- Ports: `7474` (Browser UI), `7687` (Bolt protocol).
- Auth from env vars: `${NEO4J_USER}/${NEO4J_PASSWORD}`.
- Memory tuning: 512MB heap initial, 1GB max heap, 512MB page cache.
- Enables native vector index: `dbms.security.procedures.unrestricted: "db.index.*"`.
- Persistent volumes: `./neo4j/data` and `./neo4j/logs`.
- Health check via `neo4j status` with 30s start period.

**`.env.example`** — Added: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `LLAMA_PARSE_APIKEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, `MISTRAL_API_KEY`.

---

#### **Testing**

Extensive new test suites for the RAG pipeline:

**`tests/rag/documentIngestion/`** — Ingestion pipeline tests:
- `test_parse_document.py` — LlamaParse integration tests.
- `test_graph_extraction.py` — LLM graph extraction tests.
- `test_entity_resolution.py` — Fuzzy matching and LLM disambiguation tests.
- `test_graph_persistence.py` — Canonical rewriting and payload construction tests.
- `test_id_generation.py` — Stable entity ID generation tests.
- `test_ingestion_pipeline.py` — Full pipeline integration tests.
- `test_prompt_contracts.py` — Prompt template validation.
- `chunking/` — Recursive chunking tests.
- `contextual_retrieval/` — Contextual retrieval tests.

**`tests/rag/retrieve/`** — Retrieval pipeline tests:
- `test_router.py` — Intent classification tests.
- `test_grader.py`, `test_grader_deep.py` — Chunk grading tests.
- `test_graph_agent.py` — Graph traversal tests.
- `test_rag_graph.py` — Full LangGraph pipeline tests.
- `test_retrieve_pipeline.py`, `test_retrieve_pipeline_deep.py` — End-to-end retrieval tests.
- `test_voyage_reranker.py`, `test_voyage_reranker_deep.py` — Voyage AI reranker tests.
- `test_config.py` — Configuration validation tests.

**`tests/rag/real_env_test/`** — Real environment integration tests:
- `run_real_env_test.py` — Full ingestion pipeline with real API calls.
- `run_retrieve_real_env_test.py` — Full retrieval pipeline with real Neo4j + LLM.
- `test_run_real_env_test.py` — Pytest wrapper for real env tests.

**`pyproject.toml`** — Added `integration` marker and starlette deprecation warning filter.

---

#### **Dependencies (`pyproject.toml`)**

**New dependencies added:**

| Package | Version | Purpose |
|---|---|---|
| `llama-cloud` | ≥2.0.0 | LlamaParse document parsing API |
| `transformers` | ≥4.40.0 | Jina tokenizer (`XLMRobertaTokenizerFast`) for token counting |
| `langchain-text-splitters` | ≥0.3.0 | `RecursiveCharacterTextSplitter` for document chunking |
| `rapidfuzz` | ≥3.9.0 | Fuzzy string matching for entity resolution |
| `neo4j` | ≥5.0.0 | Neo4j async driver (`AsyncGraphDatabase`) |
| `respx` | ≥0.21.0 | HTTP mocking for tests |

`structlog` moved from implicit to explicit dependency (`≥23.3.0`).
`pytest` moved from dev-only to main dependencies.

---

#### **Other Additions**

- **`data/pankajkumar.pdf`** — Sample document for ingestion testing.
- **`data/neo4j/`** — Neo4j persistent data directory.
- **`graphify-out/`** — Graphify code analysis output: `GRAPH_REPORT.md`, `graph.html`, `graph.json`, `manifest.json`.
- **`variables_review.md`** — Configuration variables review document.

---

### 📁 File Tree

```
TopicTrace/
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── variables_review.md
│
├── data/
│   ├── pankajkumar.pdf
│   └── neo4j/
│       └── data/
│
├── docs/
│   └── superpowers/
│
├── graphify-out/
│   ├── GRAPH_REPORT.md
│   ├── graph.html
│   ├── graph.json
│   ├── manifest.json
│   └── cost.json
│
├── src/
│   └── topictrace/
│       ├── __init__.py                      # Structured logging setup (structlog)
│       ├── settings.py                      # Pydantic-based centralized config (multi-LLM, Neo4j, Jina, Voyage)
│       │
│       ├── db/
│       │   ├── __init__.py
│       │   ├── neo4j/
│       │   │   ├── __init__.py              # Neo4jClient (AsyncGraphDatabase wrapper)
│       │   │   └── cypherQuerys.py          # All Cypher: vector index, chunk/entity CRUD, similarity search
│       │   └── postgres/
│       │       └── client.py                # Postgres pool, table creation, key hashing
│       │
│       ├── provider/
│       │   ├── __init__.py
│       │   ├── llm.py                       # Multi-provider ChatOpenAI factory (DeepSeek + Mistral)
│       │   ├── embedding.py                 # Jina Embeddings v2 client (768-dim vectors)
│       │   └── rerank.py                    # Voyage AI reranker (rerank-2 model)
│       │
│       ├── agents/
│       │   ├── state.py                     # ResearchState TypedDict (message list)
│       │   └── graph.py                     # LangGraph state machine (deep research: AGENT ↔ TOOLS)
│       │
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── research_agent.py            # System + user prompt builder (3 depth levels)
│       │   ├── router_intent_classifier.py  # Query → "simple" / "complex" classification prompt
│       │   ├── grader_chunk_evaluator.py    # Chunk sufficiency + fast-path answer prompt
│       │   ├── answer_generator.py          # Final answer from reranked context prompt
│       │   └── ingestion/
│       │       ├── extraction.py            # Entity + relationship extraction prompt (one-shot JSON)
│       │       └── resolution.py            # Entity disambiguation prompt (merge/reject decisions)
│       │
│       ├── tools/
│       │   ├── __init__.py                  # Re-exports web_fetch, web_search
│       │   ├── cache.py                     # Postgres cache (SHA-256 keys, 20-min TTL)
│       │   ├── web_search.py                # Tavily search (async, batch, @tool)
│       │   └── web_fetch.py                 # Jina fetch + LLM summarize + cache (@tool)
│       │
│       ├── rag/
│       │   ├── documentIngestion/
│       │   │   ├── __init__.py
│       │   │   ├── ingestion.py             # Pipeline orchestrator (6-stage document-to-graph)
│       │   │   ├── parseDocument.py         # LlamaParse document → structured pages
│       │   │   ├── contextual_retrieval.py  # LLM-generated retrieval context per chunk
│       │   │   ├── graphExtraction.py       # LLM entity + relationship extraction per chunk
│       │   │   ├── entityResolution.py      # Exact normalization → fuzzy merge → LLM disambiguation
│       │   │   ├── graphPersistence.py      # Canonical rewriting + Neo4j write payload builder
│       │   │   ├── graphRelationshipSchema.py # Allowed relationship types schema
│       │   │   ├── chunking/
│       │   │   │   ├── __init__.py          # Public API: chunk_document, count_tokens
│       │   │   │   └── recursive.py         # RecursiveCharacterTextSplitter (512 tok, 100 overlap)
│       │   │   └── models/
│       │   │       └── graphExtractionModels.py # Pydantic: Entity, Relationship, Resolution models
│       │   │
│       │   └── documentRetrieve/
│       │       ├── retrieve.py              # Pipeline entry point: handle_query()
│       │       ├── router.py                # LLM intent classifier (simple/complex)
│       │       ├── grader.py                # LLM chunk sufficiency evaluator (fast-path answer)
│       │       ├── graphAgent.py            # Neo4j 1-hop graph traversal → fact sheet
│       │       └── graph/
│       │           ├── __init__.py
│       │           ├── build_graph.py       # LangGraph compiler (6 nodes, conditional edges)
│       │           ├── state.py             # RAGState TypedDict (14 fields)
│       │           ├── nodes.py             # 6 node functions: route, search, grade, graph, rerank, answer
│       │           └── edges.py             # 2 conditional edge functions: intent routing + grader routing
│       │
│       └── server/
│           ├── __init__.py
│           ├── app.py                       # FastAPI entry point (Postgres + Neo4j init, 4 routers)
│           │
│           ├── middleware/
│           │   └── __init__.py              # Request ID, rate limit, CORS, API key auth
│           │
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── api_key.py               # POST /api-keys (generate & store)
│           │   ├── deep_research/
│           │   │   └── research.py          # POST /research, POST /research/stream (SSE)
│           │   └── rag/
│           │       ├── ingestionAPI.py       # POST /ingestion/ingest
│           │       └── retrieveAPI.py        # POST /retrieve/query
│           │
│           └── schemas/
│               ├── deep_research/
│               │   └── research.py          # ResearchRequest/Response (query + depth)
│               └── rag/
│                   ├── ingestionModels.py    # IngestionRequest/Response
│                   └── retrieveModels.py     # QueryRequest/Response (top_k, intent, graph metadata)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                          # Shared pytest fixtures
    ├── test_project_structure.py            # Project structure validation
    ├── test_deep_audit.py                   # Comprehensive system audit tests
    ├── tools/
    │   └── __init__.py                      # Tools module tests
    └── rag/
        ├── __init__.py
        ├── documentIngestion/
        │   ├── __init__.py
        │   ├── test_parse_document.py       # LlamaParse tests
        │   ├── test_graph_extraction.py     # Entity/relationship extraction tests
        │   ├── test_entity_resolution.py    # Fuzzy matching + LLM disambiguation tests
        │   ├── test_graph_persistence.py    # Canonical rewriting + payload tests
        │   ├── test_id_generation.py        # Stable entity ID tests
        │   ├── test_ingestion_pipeline.py   # Full pipeline integration tests
        │   ├── test_prompt_contracts.py     # Prompt template validation
        │   ├── chunking/                    # Recursive chunking tests
        │   └── contextual_retrieval/        # Contextual retrieval tests
        ├── retrieve/
        │   ├── __init__.py
        │   ├── test_config.py               # Configuration validation
        │   ├── test_router.py               # Intent classification tests
        │   ├── test_grader.py               # Chunk grading tests
        │   ├── test_grader_deep.py          # Deep grader edge case tests
        │   ├── test_graph_agent.py          # Graph traversal tests
        │   ├── test_rag_graph.py            # Full LangGraph pipeline tests
        │   ├── test_retrieve_pipeline.py    # End-to-end retrieval tests
        │   ├── test_retrieve_pipeline_deep.py # Deep retrieval edge cases
        │   ├── test_voyage_reranker.py      # Voyage AI reranker tests
        │   └── test_voyage_reranker_deep.py # Deep reranker edge cases
        └── real_env_test/
            ├── __init__.py
            ├── run_real_env_test.py          # Real ingestion integration test
            ├── run_retrieve_real_env_test.py # Real retrieval integration test
            └── test_run_real_env_test.py     # Pytest wrapper
```

---
---

## [0.1.0] — 2026-06-06

### 🎯 Summary

Initial release of TopicTrace — an AI-powered research agent built for exam preparation. The system accepts a student's question via a REST API, autonomously searches the web, fetches and summarizes relevant pages, caches results in PostgreSQL, and returns a structured, exam-focused answer. The entire agent loop is orchestrated by a LangGraph state machine powered by DeepSeek V4 Flash through NVIDIA's API.

---

### 🏗️ Core Architecture

The request lifecycle flows through the following chain:

```
Client Request
  → FastAPI Server (app.py)
    → Middleware Stack (request ID → rate limiter → CORS → auth)
      → Research Endpoint (research.py)
        → LangGraph State Machine (graph.py)
          → LLM decides: need more data?
            → Yes: call web_search (Tavily) → call web_fetch (Jina Reader) → LLM summarizes → cache in Postgres → loop back
            → No: return final answer
        → Response sent to client
```

---

### ✅ What Was Added

#### **Structured Logging (`src/topictrace/__init__.py`)**

Global logging configuration using `structlog`. Every log line across the entire application includes:
- **Timestamp** — formatted as `HH:MM:SS` in local time for quick visual scanning.
- **Log level** — DEBUG, INFO, WARNING, ERROR for filtering.
- **Callsite info** — the exact filename, function name, and line number that produced the log, so you never have to guess which module logged what.
- **Context variables** — structlog's `merge_contextvars` processor allows any middleware to bind key-value pairs (like `request_id`) that automatically appear in all subsequent log lines within that request, without passing the logger object around.
- **Colored console output** — `ConsoleRenderer` with forced ANSI colors for readable terminal output during development.

A single `log` object is exported and imported by every module in the project, ensuring consistent formatting everywhere.

---

#### **Centralized Configuration (`src/topictrace/settings.py`)**

All environment variables, API endpoints, and tuning constants are loaded and validated in one place using `python-dotenv`. The module:

- **Loads `.env`** at import time via `load_dotenv()`.
- **Reads three required secrets** from environment: `LLM_API_KEY` (for NVIDIA/DeepSeek), `TAVILY_API_KEY` (for web search), and `DATABASE_URL` (for PostgreSQL connection).
- **Fails fast** — if any required variable is missing, the process exits immediately with a clear error message listing exactly which variables are absent. This prevents the app from starting in a broken state.
- **Defines all constants** in one file:
  - `LLM_BASE_URL` — points to NVIDIA's OpenAI-compatible endpoint (`integrate.api.nvidia.com/v1`).
  - `LLM_MODEL` — `deepseek-ai/deepseek-v4-flash`, a fast reasoning model.
  - `JINA_READER_BASE_URL` — Jina's reader proxy (`r.jina.ai/`) that converts any URL to clean markdown.
  - `CACHE_TTL_SECONDS` — 20 minutes. Cached summaries expire after this window so the agent fetches fresh data for rapidly-changing topics.
  - `SUMMARIZE_MAX_INPUT_CHARS` — 8,000 characters. Raw page content is truncated to this length before being sent to the LLM for summarization, preventing token limit overflows.
  - `SUMMARIZE_MAX_TOKENS` — 1,024 tokens max for summaries.
  - `SUMMARIZE_TEMPERATURE` — 0.7 for balanced creativity in summaries.
  - `SEARCH_MAX_RESULTS` — 10 results per Tavily search call.
  - `SEARCH_SNIPPET_MAX_CHARS` — 300 characters per search snippet.
  - `FETCH_TIMEOUT_SECONDS` — 30 seconds before a Jina fetch request is abandoned.

---

#### **Database Layer (`src/topictrace/db/client.py`)**

PostgreSQL connection management using `psycopg` (v3) with connection pooling:

- **Connection Pool** — `psycopg_pool.ConnectionPool` with `min_size=4` and `max_size=10`. The pool pre-opens 4 connections at startup and scales up to 10 under load. When a request needs a database connection, it borrows one from the pool instead of opening a new TCP connection (which takes ~50ms). When done, the connection returns to the pool for reuse.
- **`generate_key_hash(key_part)`** — hashes the secret portion of an API key using SHA-256. The raw key is never stored in the database — only its hash. This means even if the database is compromised, the attacker cannot recover usable API keys.
- **`init_db()`** — creates two tables and two indexes on startup using `CREATE IF NOT EXISTS` (idempotent, safe to run repeatedly):

  **Table: `api_keys`**
  | Column | Type | Purpose |
  |---|---|---|
  | `id` | SERIAL PK | Auto-incrementing row identifier |
  | `key_prefix` | TEXT | The non-secret prefix (e.g., `tt`) used to identify keys without exposing them |
  | `key_hash` | TEXT UNIQUE | SHA-256 hash of the secret portion — used for authentication lookups |
  | `is_active` | BOOLEAN | Soft-delete flag. Set to FALSE to revoke a key without deleting the row |
  | `created_at` | TIMESTAMP | Auto-set to `NOW()` on insert |

  **Table: `research_cache`**
  | Column | Type | Purpose |
  |---|---|---|
  | `id` | SERIAL PK | Auto-incrementing row identifier |
  | `query_hash` | TEXT UNIQUE | SHA-256 hash of `query:url` — the cache key |
  | `result` | JSONB | The cached summarized content, stored as `{"content": "..."}` |
  | `expires_at` | TIMESTAMP | When this cache entry becomes stale (current time + 20 min TTL) |
  | `created_at` | TIMESTAMP | Auto-set to `NOW()` on insert |

  **Indexes:**
  - `idx_research_cache_lookup` on `query_hash` — makes cache lookups O(log n) instead of O(n) full table scans.
  - `idx_expires_at` on `expires_at` — speeds up TTL expiration queries.

---

#### **LLM Provider (`src/topictrace/provider/llm.py`)**

Factory functions that return a configured LangChain `ChatOpenAI` instance:

- **`get_llm()`** — creates a `ChatOpenAI` pointing at NVIDIA's API with DeepSeek V4 Flash. Key details:
  - Uses custom `httpx.Client` and `httpx.AsyncClient` with `Accept-Encoding: identity` header to disable gzip compression. This works around a known bug where NVIDIA's gateway sends broken compressed responses.
  - 60-second timeout on both sync and async HTTP clients.
  - Enables DeepSeek's thinking mode via `extra_body` with `"thinking": True` and `"reasoning_effort": "high"` — this makes the model show its chain-of-thought before answering, improving research quality.
- **`get_llm_with_tools(tools)`** — wraps `get_llm()` and calls `.bind_tools(tools)` to attach tool schemas to the model. When the LLM receives these schemas, it can emit structured tool-call messages that LangGraph's `ToolNode` knows how to execute.

---

#### **Agent State (`src/topictrace/agents/state.py`)**

Defines the `ResearchState` TypedDict — the single data structure that flows through every node of the LangGraph:

- Contains one field: `messages`, annotated with `add_messages`. This annotation tells LangGraph to **append** new messages to the list instead of replacing it. Without this annotation, each node would overwrite the entire conversation history, and the agent would lose context between steps.

---

#### **Agent Graph (`src/topictrace/agents/graph.py`)**

The LangGraph state machine that controls the agent's reasoning loop:

- **Two nodes:**
  - `AGENT` — prepends the system prompt, sends the full message history to the LLM, and returns the LLM's response (which may contain tool calls or a final answer).
  - `TOOLS` — a `ToolNode` that automatically executes any tool calls the LLM requested (`web_search` or `web_fetch`) and appends the results back to the message list.

- **Routing logic (`should_continue`):**
  - If the last message contains `tool_calls` → route to `TOOLS` (the LLM wants more data).
  - Otherwise → route to `END` (the LLM has produced its final answer).

- **Edge wiring:**
  - `START → AGENT` — every request begins at the agent node.
  - `AGENT → (conditional) → TOOLS or END` — the agent decides whether to loop.
  - `TOOLS → AGENT` — after executing tools, control always returns to the agent so it can process the results.

- The compiled `app` object is the runnable graph, invoked by the research endpoint.

---

#### **System Prompt (`src/topictrace/prompts/agent_system.py`)**

The prompt engineering layer that controls the LLM's behavior:

- **`get_system_prompt()`** — returns the agent's identity and behavioral rules:
  - Identity: "You are TopicTrace — an AI research assistant built for exam preparation."
  - Tool usage order: search first → fetch relevant URLs → summarize long pages.
  - Includes good vs. bad query examples (e.g., "physics questions" is bad, "CIE A-Level Physics 9702 Paper 4 past questions 2024" is good).
  - Output rules: cite sources, use Markdown headings, highlight exam-relevant content (formulas, dates, definitions).
  - Behavioral constraint: never answer from memory for specific papers/questions — always search.

- **`get_user_prompt(query, depth)`** — wraps the student's raw question with depth-specific research instructions. Three depth profiles:

  | Depth | Max Searches | Max Fetches | Summary Style |
  |---|---|---|---|
  | `quick` | 1 | 1 | Brief, 3–5 bullet points |
  | `standard` | 2 | 3 | Structured with headings and bullet points |
  | `deep` | 4 | 6 | Comprehensive with sections, examples, and key takeaways |

  The `deep` profile instructs the agent to search from multiple angles: syllabus, past papers, revision notes, and marking schemes.

---

#### **Web Search Tool (`src/topictrace/tools/web_search.py`)**

An async LangChain `@tool` that queries Tavily's search API:

- **Batch support** — accepts a single query string or a list of queries. Multiple queries are executed concurrently using `asyncio.gather`, so 4 searches take the same wall-clock time as 1.
- **Input validation** — rejects non-string, non-list inputs with a descriptive error message.
- **API key guard** — checks `TAVILY_API_KEY` before making any request. Returns a helpful error if the key is missing.
- **Result extraction** — each Tavily result is cleaned into a `{title, url, snippet}` dict. Snippets are truncated to 300 characters to keep the agent's context window lean.
- **Error isolation** — if one query in a batch fails, the others still succeed. Failed queries get an error entry in the results list instead of crashing the entire batch.
- **Logging** — logs success count and total count for every batch via structlog.

---

#### **Web Fetch Tool (`src/topictrace/tools/web_fetch.py`)**

An async LangChain `@tool` that downloads web pages via Jina Reader and summarizes them with the LLM:

- **Cache-first strategy** — before fetching any URL, the tool generates a SHA-256 cache key from `query:url` and checks PostgreSQL. If a cached summary exists and hasn't expired (20-min TTL), it returns immediately — no HTTP request, no LLM call.
- **Parallel fetching** — uncached URLs are fetched concurrently via `httpx.AsyncClient` with `asyncio.gather`. Jina Reader converts raw HTML into clean Markdown.
- **LLM summarization** — each fetched page is summarized by calling the LLM with a focused prompt: "Summarize the provided content in relation to the user's query. Be concise, factual, and focus on exam-relevant information." The raw page content is truncated to 8,000 characters before sending.
- **Cache write-back** — after summarization, the result is saved to PostgreSQL with a 20-minute TTL. The next request for the same query+URL combination gets an instant cache hit.
- **Error handling** — handles network errors, non-200 status codes (403, 404, 451), empty URLs, and unexpected exceptions. Each error produces a descriptive entry in the results list without crashing the batch.

---

#### **Cache System (`src/topictrace/tools/cache.py`)**

PostgreSQL-backed cache with TTL expiration:

- **`generate_fetch_cache_key(query, url)`** — creates a deterministic cache key by hashing `query:url` (lowercased and stripped) with SHA-256, prefixed with `cache:tool:web_fetch:`. The same URL with different queries produces different cache keys, so a page summarized for "physics" won't be returned for "chemistry."
- **`load_from_cache(cache_key)`** — queries `research_cache` table. Only returns results where `expires_at > NOW()`. Returns `None` on miss or any database error (fail-open: if the cache breaks, the system fetches fresh data instead of crashing).
- **`save_to_cache(cache_key, result, ttl)`** — inserts or updates (upsert via `ON CONFLICT DO UPDATE`) the cache entry. The result is wrapped in `{"content": "..."}` and stored as JSONB. Expiration is calculated as `NOW() + ttl` seconds.

---

#### **FastAPI Server (`src/topictrace/server/app.py`)**

The application entry point:

- **Lifespan handler** — runs `init_db()` on startup to create tables if they don't exist. Uses FastAPI's `asynccontextmanager`-based lifespan (the modern replacement for deprecated `on_startup` events).
- **Router registration** — mounts `research_router` and `api_key_router`.
- **Health check** — `GET /health/live` returns `{"status": "ok"}`. Used by Docker's `HEALTHCHECK` and load balancers.
- **Middleware import** — imports the middleware module *after* `app` is created, triggering `@app.middleware` decorator registration.

---

#### **Middleware Stack (`src/topictrace/server/middleware/__init__.py`)**

Four middleware layers executed in order on every request:

1. **Request ID Tracing** — generates a unique 12-character hex ID (or uses the incoming `X-Request-ID` header if present). Binds it to structlog's context variables so every log line in that request automatically includes the ID. Returns the ID in the response header for client-side correlation.

2. **Rate Limiting** — sliding window rate limiter: 10 requests per minute per IP address. Uses an in-memory `defaultdict(list)` of timestamps. On each request, expired entries (older than 60 seconds) are pruned, and if 10+ entries remain, a `429 Too Many Requests` is returned.

3. **CORS** — allows all origins, methods, and headers (`*`). Permissive for development; should be locked down in production.

4. **API Key Authentication** — skips auth for `/health/live` and `/api-keys`. For all other paths:
   - Extracts `Bearer <key>` from the `Authorization` header.
   - Splits the key at `_` into `prefix` and `secret`.
   - Hashes the secret with SHA-256 and queries `api_keys` table for a matching active row.
   - Returns `401` with specific error messages for missing header, empty key, invalid format, or unrecognized key.

---

#### **API Routes**

**`POST /research` (`src/topictrace/server/routes/research.py`)**
- Accepts a JSON body: `{"query": "...", "depth": "quick|standard|deep"}`.
- Builds a `HumanMessage` with the depth-framed user prompt.
- Invokes the LangGraph agent asynchronously via `app.ainvoke(state)`.
- Returns the last message content as `{"answer": "..."}`.

**`POST /research/stream` (`src/topictrace/server/routes/research.py`)**
- Same input as `/research`.
- Returns a `StreamingResponse` with `text/event-stream` media type (Server-Sent Events).
- Uses `app.astream_events(state, version="v2")` to stream LLM tokens in real time.
- Each token is sent as `data: {"token": "..."}`.
- Tool calls emit `data: {"status": "toolCalling", "tool": "web_search"}`.
- Errors emit `data: {"status": "error", "error": "..."}`.
- Stream ends with `data: [DONE]`.

**`POST /api-keys` (`src/topictrace/server/routes/api_key.py`)**
- Generates a cryptographically secure API key using `secrets.token_urlsafe(32)`.
- Key format: `tt_<random-base64-string>`.
- Stores only the hash of the random portion in the database. The full key is returned to the caller once and never stored in plaintext.

---

#### **Request/Response Schemas (`src/topictrace/server/schemas/research.py`)**

Pydantic models for input validation:

- **`ResearchRequest`** — `query: str` (required), `depth: Literal["quick", "standard", "deep"]` (defaults to `"quick"`). FastAPI auto-validates incoming JSON against this schema and returns a `422` with field-level errors if validation fails.
- **`ResearchResponse`** — `answer: str`. Ensures a consistent response shape.

---

#### **Docker & Deployment**

**`Dockerfile`**
- Base image: `python:3.11`.
- Installs `curl` for health checks.
- Uses `uv` for fast dependency resolution and installation (`pip install uv && uv sync`).
- Runs uvicorn on port 8080 with `--reload` for development hot-reloading.
- Includes a `HEALTHCHECK` that pings `/health/live` every 30 seconds.

**`docker-compose.yml`**
- **`postgres` service** — PostgreSQL 16 with a persistent named volume (`pgdata`). Health-checked via `pg_isready`. Database name: `topictrace_db`, user: `topictrace`.
- **`topictrace` service** — builds from `Dockerfile`. Mounts `src/` and `pyproject.toml` as volumes for live code reloading. Depends on `postgres` with `condition: service_healthy` — the API container won't start until Postgres is ready to accept connections.

**`.env.example`**
- Template for required environment variables: `LLM_API_KEY`, `TAVILY_API_KEY`. Copy to `.env` and fill in values.

---

#### **Testing**

- **`tests/conftest.py`** — shared pytest fixtures.
- **`tests/test_project_structure.py`** — validates that the project file structure matches expectations.
- **`tests/test_deep_audit.py`** — comprehensive audit tests covering the full system.
- **`tests/tools/`** — isolated tests for the tools module.

---

#### **Dependencies (`pyproject.toml`)**

| Package | Version | Purpose |
|---|---|---|
| `tavily-python` | ≥0.5.0 | Web search API client |
| `structlog` | latest | Structured logging with context variables |
| `requests` | ≥2.31.0 | HTTP client (general purpose) |
| `openai` | ≥1.50.0 | OpenAI API client (used indirectly via LangChain) |
| `python-dotenv` | ≥1.0.0 | Load `.env` files into `os.environ` |
| `rich` | ≥13.0.0 | Rich terminal formatting |
| `prompt-toolkit` | ≥3.0.0 | Interactive terminal prompts |
| `httpx` | ≥0.27.0 | Async HTTP client for Jina Reader fetching |
| `langgraph` | ≥1.2.2 | State machine framework for the agent loop |
| `langchain-core` | ≥1.4.0 | Core abstractions: tools, messages, prompts |
| `langchain-openai` | ≥1.2.2 | OpenAI-compatible LLM provider for LangChain |
| `fastapi` | ≥0.136.3 | ASGI web framework |
| `uvicorn` | ≥0.48.0 | ASGI server |
| `psycopg[binary]` | ≥3.3.4 | PostgreSQL adapter (v3, async-capable) |
| `psycopg-pool` | ≥3.3.1 | Connection pooling for psycopg |

**Dev dependencies:** `pytest` ≥8.0.0, `pytest-cov` ≥5.0.0.

**Build system:** Hatchling. Wheel packages built from `src/topictrace`.
