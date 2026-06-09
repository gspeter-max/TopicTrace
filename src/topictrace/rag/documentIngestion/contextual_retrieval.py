"""
Contextual retrieval pipeline.

This module parses a document, chunks the full text, generates a short
retrieval-oriented summary for each chunk, and returns contextualized chunks.
It does not embed text or write vectors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from topictrace import settings
from topictrace.rag.documentIngestion.chunking import chunk_document
from topictrace.rag.documentIngestion.parseDocument import (
    get_all_pages_text,
    parse_document,
)


async def build_context_messages(
    full_document_text: str, chunk: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write concise retrieval context for document chunks. "
                "Your job is to explain where this chunk fits in the larger document."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document id: {chunk['document_id']}\n"
                f"Chunk index: {chunk['chunk_index']}\n"
                f"Chunk token count: {chunk['token_count']}\n\n"
                f"Full document text:\n{full_document_text}\n\n"
                f"Chunk text:\n{chunk['text']}\n\n"
                "Write 1 to 3 short sentences that help retrieval. "
                "Mention the section, topic, or document role of this chunk. "
                "Do not summarize the chunk alone. Return plain text only."
            ),
        },
    ]


async def generate_chunk_context(
    *,
    client: BaseChatModel,
    full_document_text: str,
    chunk: dict[str, Any],
    model: str | None = None,
    max_tokens: int = settings.CONTEXTUAL_RETRIEVAL_MAX_TOKENS,
) -> dict[str, Any]:
    messages = await build_context_messages(full_document_text, chunk)
    bind_kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
    if model:
        bind_kwargs["model"] = model
    bound_client = client.bind(**bind_kwargs)
    response = await bound_client.ainvoke(messages)
    context = (response.content or "").strip()
    contextualized_text = f"{context}\n\n{chunk['text']}"
    return {
        **chunk,
        "context": context,
        "contextualized_text": contextualized_text,
        "model": model or getattr(client, "model_name", "unknown"),
    }


async def build_contextualized_document(
    *,
    file_path: str,
    client: BaseChatModel,
    model: str | None = None,
    max_concurrency=settings.CONTEXTUAL_RETRIEVAL_MAX_CONCURRENCY,
) -> dict[str, Any]:
    parsed = parse_document(file_path)
    full_document_text = get_all_pages_text(parsed)
    document_id = Path(file_path).name
    chunks = chunk_document(full_document_text, document_id=document_id)

    sem = asyncio.Semaphore(max_concurrency)

    async def _run_one(chunk: dict[str, Any]):
        async with sem:
            return await generate_chunk_context(
                client=client,
                full_document_text=full_document_text,
                chunk=chunk,
                model=model,
            )

    tasks = [_run_one(chunk) for chunk in chunks]
    contextualized_chunks = await asyncio.gather(*tasks)

    return {
        "document_id": document_id,
        "source_file": file_path,
        "full_document_text": full_document_text,
        "chunks": contextualized_chunks,
        "chunk_count": len(contextualized_chunks),
        "model": model,
    }
