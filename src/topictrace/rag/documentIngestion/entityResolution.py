import json
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from rapidfuzz import fuzz, process

from topictrace import settings
from topictrace.prompts.ingestion.resolution import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from topictrace.rag.documentIngestion.models.graphExtractionModels import (
    EntityResolutionDecision,
)


def find_fuzzy_merge_candidates(
    raw_entity_names: set[str],
    threshold: int = settings.ENTITY_RESOLUTION_FUZZY_THRESHOLD,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Finds pairs of similar names (e.g. 'Apple' vs 'Apple Inc')
    and returns unmatched names separately."""

    entity_list = list(raw_entity_names)

    similarity_matrix = process.cdist(
        entity_list,
        entity_list,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,  # skips storing scores below threshold early → faster
        workers=-1,  # uses all CPU cores → faster on large inputs
    )

    ROWS, COLS = np.where(similarity_matrix >= threshold)
    fuzzy_candidates = []
    used_candidates = set()

    for r, c in zip(ROWS, COLS):
        if r < c:
            fuzzy_candidates.append((entity_list[r], entity_list[c]))
            used_candidates.add(entity_list[r])
            used_candidates.add(entity_list[c])

    left_candidates = raw_entity_names.difference(used_candidates)
    return fuzzy_candidates, left_candidates


def split_clear_cases_from_ambiguous_cases(
    pairs_with_scores: list[tuple[str, str, float]],
    high_threshold: float = settings.ENTITY_RESOLUTION_HIGH_THRESHOLD,
) -> tuple[list[tuple[str, str, float]], set[str]]:
    """This function checks that we can correctly separate the pairs of names that are super obvious \
        from the pairs that are confusing and need the AI to look at them."""

    same_entity_pairs = []
    different_entity = set()

    for left, right, score in pairs_with_scores:
        if score >= high_threshold:
            same_entity_pairs.append((left, right, score))
        else:
            different_entity.add(left)
            different_entity.add(right)

    return same_entity_pairs, different_entity


def build_entity_resolution_messages(entity_names: set[str]) -> list[dict[str, Any]]:
    """Build messages containing the flat list of entity names for the LLM to resolve."""
    payload = {"entity_names": list(entity_names)}

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                payload_json=json.dumps(payload, indent=2)
            ),
        },
    ]


async def resolve_ambiguous_entity_pairs(
    *,
    llm_client: BaseChatModel,
    ambiguous_pairs: set[str],
    model: str | None = None,
) -> list[EntityResolutionDecision]:
    """This function asks the AI to look at pairs of names we are confused about, and decide if they are the same thing or different things."""
    if not ambiguous_pairs:
        return []

    bind_kwargs = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    if model:
        bind_kwargs["model"] = model
    bound_client = llm_client.bind(**bind_kwargs)
    response = await bound_client.ainvoke(
        build_entity_resolution_messages(ambiguous_pairs)
    )
    content = response.content
    content_str = content if isinstance(content, str) else ""
    response_payload = json.loads(content_str or "{}")

    return [
        EntityResolutionDecision(**decision_row)
        for decision_row in response_payload.get("decisions", [])
    ]
