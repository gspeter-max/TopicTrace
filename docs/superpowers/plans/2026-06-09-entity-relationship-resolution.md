# Entity and Relationship Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **code-change** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal information for freshAgent** :
- Fix two bugs in the TopicTrace ingestion pipeline:
  1. Relationship endpoints (`source_entity_name`, `target_entity_name`) are not resolved to their canonical names, causing broken edges in Neo4j.
  2. Unmapped entities (entities not present in the canonical name map) are incorrectly skipped instead of falling back to their raw names.
- Files to read to understand the code:
  - [ingestion.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentIngestion/ingestion.py) - Ingestion orchestration and `resolve_entities_for_graph` definition.
  - [graphPersistence.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentIngestion/graphPersistence.py) - Mapping of raw results to canonical entities/relationships.
  - [test_graph_persistence.py](file:///Users/apple/project/TopicTrace/tests/rag/documentIngestion/test_graph_persistence.py) - Existing tests that verify rewriting logic.

**Architecture:**
- `resolve_entities_for_graph` returns a pre-computed alias-to-canonical dictionary (`canonical_by_alias`) mapping both canonical names and alias names to the canonical name.
- `rewrite_graph_results_to_canonical_entities` uses this mapping:
  - For entities: if `canonical_by_alias[name] == name`, it is canonical (keep). If `canonical_by_alias[name] != name`, it is an alias (skip). If `name` is not mapped, keep as-is.
  - For relationships: resolve both endpoints using `canonical_by_alias` and deduplicate within the chunk/document to avoid redundant relationships.

**Important Rule to follow :**
- **CRITICAL:** Make the code function name and variable name clear and easily understandable instead of short and confusing names so a junior developer or reader can instantly comprehend.
- **Explain like a fresher:** Make all comments and docs human-readable, literal, and step-by-step.

---


### Task 2: Refactor `resolve_entities_for_graph` in `ingestion.py`
**Files:**
- Modify: [ingestion.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentIngestion/ingestion.py)

- [ ] **Step 1: Update type annotations and return value**
  Modify `resolve_entities_for_graph` to build and return `canonical_by_alias`.
  
  ```python
  async def resolve_entities_for_graph(
      raw_chunk_graph_results: list[Any], 
      existing_entity: set[str],
      llm_provider: Literal["MISTRAL_AI", "DEEPSEEK_AI"] = settings.DEFAULT_LLM_PROVIDER
  ) -> dict[str, dict[str, Any]]:
      """
      Collapses raw entity name variants into one canonical name per real-world entity.
      (Existing implementation...)
      """
      # ... (existing canonical_map generation logic) ...

      # Build the alias-to-canonical lookup map
      canonical_by_alias: dict[str, str] = {}
      for canonical_name, raw_names in canonical_map.items():
          canonical_by_alias[canonical_name] = canonical_name
          for raw_name in raw_names:
              canonical_by_alias[raw_name] = canonical_name

      return {
          "canonical_name_by_raw_name": canonical_map,
          "canonical_by_alias": canonical_by_alias,
      }
  ```

- [ ] **Step 2: Update the caller in `ingest_document_graph`**
  Modify `ingest_document_graph` in `ingestion.py` to extract `canonical_name_by_raw_name` and pass both maps to `rewrite_graph_results_to_canonical_entities`.
  
  ```python
  # In ingest_document_graph:
  resolution_result = await resolve_entities_for_graph(
      raw_results, 
      existing_entity=neo4j_entitys
  )
  
  canonical_payload = rewrite_graph_results_to_canonical_entities(
      raw_results,
      resolution_result["canonical_name_by_raw_name"],
      canonical_by_alias=resolution_result.get("canonical_by_alias")
  )
  ```

### Task 3: Refactor `rewrite_graph_results_to_canonical_entities` in `graphPersistence.py`
**Files:**
- Modify: [graphPersistence.py](file:///Users/apple/project/TopicTrace/src/topictrace/rag/documentIngestion/graphPersistence.py)

- [ ] **Step 1: Implement the canonical lookup and fallback logic**
  Update the signature and implementation of `rewrite_graph_results_to_canonical_entities` to cleanly rewrite entities and relationship endpoints.
  
  ```python
  def rewrite_graph_results_to_canonical_entities(
      raw_chunk_graph_results: list[ChunkGraphExtractionResult],
      canonical_name_by_raw_name: dict[str, list[str]],
      canonical_by_alias: dict[str, str] | None = None,
  ) -> CanonicalGraphPersistencePayload:
      """
      Rewrites every raw entity and relationship name to its resolved canonical name.
      """
      if not isinstance(canonical_name_by_raw_name, dict):
          raise TypeError(
              f"canonical_name_by_raw_name must be a dict, got {type(canonical_name_by_raw_name)!r}"
          )

      # If lookup map is not pre-computed, build it dynamically (ensures backward compatibility/test safety)
      if canonical_by_alias is None:
          canonical_by_alias = {}
          for canonical, aliases in canonical_name_by_raw_name.items():
              canonical_by_alias[canonical] = canonical
              for alias in aliases:
                  canonical_by_alias[alias] = canonical

      rewritten_entities: list[dict[str, Any]] = []
      rewritten_relationships: list[dict[str, Any]] = []
      seen_relationships: set[tuple[str, str, str]] = set()

      for chunk_entitys_relationships in raw_chunk_graph_results:
          for entity in chunk_entitys_relationships.entities:
              entity_name = entity.entity_name
              resolved_canonical = canonical_by_alias.get(entity_name)

              # If not mapped at all, treat it as a new/unmapped canonical entity (keep)
              if resolved_canonical is None:
                  rewritten_entities.append(
                      {
                          "canonical_name": entity_name,
                          "entity_type": entity.entity_type,
                          "chunk_id": entity.chunk_id,
                          "evidence_text": entity.evidence_text,
                          "alias": []
                      }
                  )
              # If mapped and resolved name equals original, it is canonical (keep)
              elif resolved_canonical == entity_name:
                  rewritten_entities.append(
                      {
                          "canonical_name": entity_name,
                          "entity_type": entity.entity_type,
                          "chunk_id": entity.chunk_id,
                          "evidence_text": entity.evidence_text,
                          "alias": canonical_name_by_raw_name.get(entity_name, [])
                      }
                  )
              # If mapped and resolved name is different, it is an alias (skip)
              else:
                  continue

          for relationship in chunk_entitys_relationships.relationships:
              # Resolve source and target to their canonical names (fallback to raw if unmapped)
              source_canonical = canonical_by_alias.get(relationship.source_entity_name, relationship.source_entity_name)
              target_canonical = canonical_by_alias.get(relationship.target_entity_name, relationship.target_entity_name)

              relationship_key = (source_canonical, relationship.relationship_type, target_canonical)
              if relationship_key in seen_relationships:
                  continue
              seen_relationships.add(relationship_key)

              rewritten_relationships.append(
                  {
                      "source_entity_name": source_canonical,
                      "relationship_type": relationship.relationship_type,
                      "target_entity_name": target_canonical,
                      "chunk_id": relationship.chunk_id,
                      "evidence_text": relationship.evidence_text,
                  }
                )

      log.debug(
          "Rewrote %d entities and %d relationships to canonical names.",
          len(rewritten_entities),
          len(rewritten_relationships),
      )
      return CanonicalGraphPersistencePayload(
          entities=rewritten_entities, relationships=rewritten_relationships
      )
  ```

### Task 4: Run tests and verify results
- [ ] **Step 2: Run graph persistence tests**
  Run: `pytest tests/rag/documentIngestion/test_graph_persistence.py -v`
  Expected: All 4 tests PASS.
  
- [ ] **Step 3: Run all ingestion tests**
  Run: `pytest tests/rag/documentIngestion/ -v`
  Expected: All ingestion tests PASS.

- [ ] **Step 4: Commit changes**
  Run:
  ```bash
  git add src/topictrace/rag/documentIngestion/ingestion.py src/topictrace/rag/documentIngestion/graphPersistence.py
  git commit -m "refactor: implement alias-to-canonical lookup and fix relationship and entity resolution"
  ```
