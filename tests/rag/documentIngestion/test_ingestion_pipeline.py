import asyncio

from topictrace.rag.documentIngestion.ingestion import ingest_document_graph


def _common_monkeypatches(monkeypatch, recorded_stage_names, doc_return):
    """Shared monkeypatching for all pipeline stage tests."""

    async def fake_build_contextualized_document(**kwargs):
        recorded_stage_names.append("context")
        return doc_return

    async def fake_build_contextualized_chunk_embeddings(chunks):
        recorded_stage_names.append("embed")
        return [[0.1, 0.2, 0.3]] * len(chunks)

    async def fake_extract_chunk_graph_data_in_parallel(*args, **kwargs):
        recorded_stage_names.append("extract")
        return []

    async def fake_get_neo4j_entities_by_document(*args, **kwargs):
        return set()

    async def fake_resolve_entities_for_graph(*args, **kwargs):
        recorded_stage_names.append("resolve")
        return {"canonical_name_by_raw_name": {}}

    async def fake_persist_document_graph(*args, **kwargs):
        recorded_stage_names.append("persist")

    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.build_contextualized_document",
        fake_build_contextualized_document,
    )
    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.build_contextualized_chunk_embeddings",
        fake_build_contextualized_chunk_embeddings,
    )
    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.extract_chunk_graph_data_in_parallel",
        fake_extract_chunk_graph_data_in_parallel,
    )
    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.get_neo4j_entities_by_document",
        fake_get_neo4j_entities_by_document,
    )
    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.resolve_entities_for_graph",
        fake_resolve_entities_for_graph,
    )
    monkeypatch.setattr(
        "topictrace.rag.documentIngestion.ingestion.persist_document_graph",
        fake_persist_document_graph,
    )


def test_ingest_document_graph_runs_all_pipeline_stages(monkeypatch):
    """All pipeline stages must fire in order: context → embed → extract → resolve → persist."""
    recorded_stage_names = []
    doc = {
        "document_id": "resume.pdf",
        "source_file": "data/resume.pdf",
        "chunks": [
            {
                "chunk_id": "resume.pdf::0",
                "chunk_index": 0,
                "text": "Alice works at Neo4j.",
                "context": "Work section",
                "contextualized_text": "Work section\n\nAlice works at Neo4j.",
            }
        ],
    }
    _common_monkeypatches(monkeypatch, recorded_stage_names, doc)

    asyncio.run(ingest_document_graph(file_path="data/resume.pdf"))

    assert recorded_stage_names == ["context", "embed", "extract", "resolve", "persist"]


def test_ingest_document_graph_handles_empty_chunks(monkeypatch):
    """Pipeline must complete and return zero counts when document has no chunks."""
    recorded_stage_names = []
    doc = {
        "document_id": "empty.pdf",
        "source_file": "data/empty.pdf",
        "chunks": [],
    }
    _common_monkeypatches(monkeypatch, recorded_stage_names, doc)

    summary = asyncio.run(ingest_document_graph(file_path="data/empty.pdf"))

    assert recorded_stage_names == ["context", "embed", "extract", "resolve", "persist"]
    assert summary["raw_entity_count"] == 0
