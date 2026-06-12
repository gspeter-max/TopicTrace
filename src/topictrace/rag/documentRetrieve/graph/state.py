"""
RAG pipeline state definition.

One TypedDict that flows through the entire LangGraph.
Every node reads fields it needs and returns only the fields it modifies.
"""

from typing import Any

from pydantic import BaseModel


class RAGState(BaseModel):
    # ── Input (set before graph.ainvoke) ──────────────────────────────────────
    query: str
    top_k: int
    top_k_rerank: int

    # ── Set by route_query node ───────────────────────────────────────────────
    intent: str  # "simple" or "complex"

    # ── Set by vector_search node ─────────────────────────────────────────────
    raw_chunks: list[dict[str, Any]]  # Full chunk dicts from Neo4j
    vector_texts: list[str]  # Extracted full_context strings

    # ── Set by grade_chunks node (simple path only) ───────────────────────────
    grade_sufficient: bool
    grade_reason: str
    grade_answer: str  # Pre-generated answer when sufficient=True

    # ── Set by graph_search node ──────────────────────────────────────────────
    graph_facts: str
    used_graph_search: bool
    reason_for_graph_search: str

    # ── Set by rerank node ────────────────────────────────────────────────────
    final_context: list[str]

    # ── Set by answer_node ────────────────────────────────────────────────────
    answer: str
