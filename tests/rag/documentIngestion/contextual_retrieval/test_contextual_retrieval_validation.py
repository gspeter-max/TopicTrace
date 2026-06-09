import asyncio

import pytest
import respx
from httpx import Response
from langchain_core.messages import AIMessage
from openai import APIStatusError

import topictrace.rag.documentIngestion.contextual_retrieval as module
from topictrace.provider.llm import get_llm
from topictrace.rag.documentIngestion.contextual_retrieval import (
    build_contextualized_document,
    generate_chunk_context,
)


class FakeBoundClient:
    def __init__(self, content: str, model_name: str = "test-model"):
        self.content = content
        self.model_name = model_name

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.content)


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.model_name = "test-model"

    def bind(self, **kwargs):
        return FakeBoundClient(
            self.content, model_name=kwargs.get("model", self.model_name)
        )


def test_validation_happy_path():
    client = FakeClient("This chunk describes the work experience section.")
    chunk = {
        "chunk_index": 0,
        "text": "Experience: built search pipelines.",
        "token_count": 6,
        "document_id": "resume.pdf",
    }

    result = asyncio.run(
        generate_chunk_context(
            client=client,
            full_document_text="full document text",
            chunk=chunk,
            model="mistral-small-latest",
        )
    )

    assert result["context"] == "This chunk describes the work experience section."
    assert result["contextualized_text"].startswith(
        "This chunk describes the work experience section."
    )


def test_validation_invalid_input_is_captured():
    client = FakeClient("irrelevant")

    with pytest.raises(KeyError):
        asyncio.run(
            generate_chunk_context(
                client=client,
                full_document_text="full document text",
                chunk={"text": "missing metadata"},
                model="mistral-small-latest",
            )
        )


def test_validation_timeout_is_captured():
    class TimeoutClient:
        def bind(self, **kwargs):
            return self

        async def ainvoke(self, *args, **kwargs):
            raise TimeoutError("request timed out")

    chunk = {
        "chunk_index": 1,
        "text": "Skills: Python, SQL, and search infrastructure.",
        "token_count": 8,
        "document_id": "resume.pdf",
    }

    with pytest.raises(TimeoutError):
        asyncio.run(
            generate_chunk_context(
                client=TimeoutClient(),
                full_document_text="full document text",
                chunk=chunk,
                model="mistral-small-latest",
            )
        )


@respx.mock
def test_validation_api_failure_is_captured(monkeypatch):
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            402,
            json={
                "error": {
                    "message": "insufficient credits",
                    "type": "billing_error",
                }
            },
        )
    )

    monkeypatch.setattr(
        "topictrace.settings.LLM_CONFIG.MISTRAL_AI.LLM_API_KEY", "test-key"
    )
    client = get_llm("MISTRAL_AI")
    chunk = {
        "chunk_index": 2,
        "text": "Education and certification details.",
        "token_count": 5,
        "document_id": "resume.pdf",
    }

    with pytest.raises(APIStatusError):
        asyncio.run(
            generate_chunk_context(
                client=client,
                full_document_text="full document text",
                chunk=chunk,
                model="mistral-small-latest",
            )
        )


def test_validation_empty_document_is_captured(monkeypatch):
    from topictrace.rag.documentIngestion import contextual_retrieval as module

    monkeypatch.setattr(module, "parse_document", lambda file_path: [{"pages": []}])
    monkeypatch.setattr(module, "get_all_pages_text", lambda parsed: "")
    monkeypatch.setattr(
        module,
        "chunk_document",
        lambda text, document_id, chunk_size=512, chunk_overlap=100: [],
    )

    client = FakeClient("unused")
    result = asyncio.run(
        build_contextualized_document(
            file_path="data/sample.pdf",
            client=client,
            model="mistral-small-latest",
        )
    )

    assert result["chunks"] == []
    assert result["chunk_count"] == 0


def test_validation_expected_load_is_captured(monkeypatch):
    monkeypatch.setattr(
        module,
        "parse_document",
        lambda file_path: [
            {"pages": [{"page": 1, "text": "page text", "md": "page md"}]}
        ],
    )
    monkeypatch.setattr(module, "get_all_pages_text", lambda parsed: "page text")
    monkeypatch.setattr(
        module,
        "chunk_document",
        lambda text, document_id, chunk_size=512, chunk_overlap=100: [
            {
                "chunk_index": i,
                "text": f"chunk {i}",
                "token_count": 2,
                "document_id": document_id,
            }
            for i in range(100)
        ],
    )

    client = FakeClient("load context")
    result = asyncio.run(
        build_contextualized_document(
            file_path="data/sample.pdf",
            client=client,
            model="mistral-small-latest",
        )
    )

    assert result["chunk_count"] == 100
    assert len(result["chunks"]) == 100
    assert result["chunks"][99]["chunk_index"] == 99


def test_validation_llm_requires_key_when_config_is_empty(monkeypatch):
    monkeypatch.setattr("topictrace.settings.LLM_CONFIG.MISTRAL_AI.LLM_API_KEY", "")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises((ValueError, Exception)):
        get_llm("MISTRAL_AI")
