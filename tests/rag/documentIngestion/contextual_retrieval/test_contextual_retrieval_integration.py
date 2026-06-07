import asyncio
import pytest
import respx
from httpx import Response
from openai import APIStatusError

from topictrace import settings
from topictrace.rag.documentIngestion.contextual_retrieval import (
    generate_chunk_context,
)
from topictrace.provider.llm import build_mistral_client


@respx.mock
def test_generate_chunk_context_with_realistic_openai_shape():
    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mistral-small-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This chunk is part of the work experience section.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    client = asyncio.run(build_mistral_client(api_key="test-key"))
    chunk = {
        "chunk_index": 0,
        "text": "Built search pipelines.",
        "token_count": 3,
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

    assert route.called
    assert result["context"] == "This chunk is part of the work experience section."
    assert result["contextualized_text"].startswith("This chunk is part of the work experience section.")


@pytest.mark.integration
def test_real_mistral_smoke():
    api_key = settings.LLM_CONFIG.MISTRAL_AI.LLM_API_KEY
    if not api_key or api_key.lower() in {"test", "test-key", "dummy", "placeholder", "missing_key", "mistral_api_key"}:
        pytest.skip("MISTRAL_API_KEY is not configured with a real value in settings")

    client = asyncio.run(build_mistral_client())

    chunk = {
        "chunk_index": 0,
        "text": "Experience: built and operated search systems.",
        "token_count": 7,
        "document_id": "resume.pdf",
    }

    try:
        result = asyncio.run(
            generate_chunk_context(
                client=client,
                full_document_text="Full document text goes here.",
                chunk=chunk,
                model="mistral-small-latest",
            )
        )

        assert isinstance(result["context"], str)
        assert result["context"].strip() != ""
    except APIStatusError as exc:
        if getattr(exc, "status_code", None) == 402:
            pytest.skip("Mistral account has no credits for the live smoke test")
        raise
