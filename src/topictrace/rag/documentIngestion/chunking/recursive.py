"""
Paragraph-aware, token-counted text chunking.

This module only splits text into chunks and counts tokens. It does not
call LLMs, embed text, or read files from disk.
"""

from __future__ import annotations

from functools import lru_cache
from topictrace import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import XLMRobertaTokenizerFast

@lru_cache(maxsize=1)
def _get_tokenizer() -> XLMRobertaTokenizerFast:
    return XLMRobertaTokenizerFast.from_pretrained(settings.TOKENIZER_MODEL)


def count_tokens(text: str) -> int:
    """Count tokens using the Jina v3 tokenizer."""
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def chunk_document(
    text: str,
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split a document string into overlapping, paragraph-aware chunks."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for index, chunk_text in enumerate(splitter.split_text(text)):
        chunks.append(
            {
                "chunk_id": f"{document_id}::{index}",
                "chunk_index": index,
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "document_id": document_id,
            }
        )
    return chunks
