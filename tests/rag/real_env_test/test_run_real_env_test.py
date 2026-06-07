import asyncio
import json

from tests.rag.real_env_test.run_real_env_test import run_pipeline_step_by_step


def test_smoke_runner_writes_partial_report_when_graph_extraction_fails(monkeypatch, tmp_path):
    async def fake_build_mistral_client():
        return object()

    async def fake_build_contextualized_document(*, file_path, client):
        return {
            "document_id": "resume.pdf",
            "source_file": str(file_path),
            "chunks": [
                {
                    "chunk_id": "resume.pdf::0",
                    "chunk_index": 0,
                    "text": "Alice sends data to Neo4j.",
                    "context": "Work section",
                    "contextualized_text": "Work section\n\nAlice sends data to Neo4j.",
                }
            ],
        }

    async def fake_build_contextualized_chunk_embeddings(chunks):
        return [[0.1]]

    async def fake_extract_chunk_graph_data_in_parallel(*args, **kwargs):
        raise ValueError("unsupported relationship type: SENDS_TO")

    monkeypatch.setattr("tests.rag.real_env_test.run_real_env_test.build_mistral_client", fake_build_mistral_client)
    monkeypatch.setattr("tests.rag.real_env_test.run_real_env_test.build_contextualized_document", fake_build_contextualized_document)
    monkeypatch.setattr("tests.rag.real_env_test.run_real_env_test.build_contextualized_chunk_embeddings", fake_build_contextualized_chunk_embeddings)
    monkeypatch.setattr("tests.rag.real_env_test.run_real_env_test.extract_chunk_graph_data_in_parallel", fake_extract_chunk_graph_data_in_parallel)

    report_path = asyncio.run(run_pipeline_step_by_step(tmp_path / "resume.pdf", tmp_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "error"
    assert report["failed_stage"] == "graph_extraction"
    assert "SENDS_TO" in report["error_message"]
    assert report["document_id"] == "resume.pdf"
    assert report["chunk_count"] == 1
    assert report["relationship_count"] == 0
    assert report["output_path"] == str(report_path)
