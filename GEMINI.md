# TopicTrace - Gemini Context

This file serves as the entry point context for the Gemini assistant.

## Project Changelog & Architecture
For the full architecture, dependency table, and recent changes, please refer directly to the project changelog:
- [CHANGELOG.md](file:///Users/apple/project/TopicTrace/CHANGELOG.md)

## Core Components
- **Ingestion Pipeline**: [ingestion.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentIngestion/ingestion.py)
- **Neo4j Queries**: [cypherQuerys.py](file:///Users/apple/project/TopicTrace/src/topictrace/db/neo4j/cypherQuerys.py)
- **Retrieval Entry**: [retrieve.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentRetrieve/retrieve.py)


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

## Development & Linting Guidelines
Before committing any changes, you MUST run type checking and linting/formatting:

1. **Pyright Type Checking**:
   ```bash
   .venv/bin/pyright
   ```

2. **Ruff Linting & Formatting**:
   ```bash
   .venv/bin/ruff check . --fix
   .venv/bin/ruff format .
   ```