import json
from typing import Any
from langchain_core.language_models import BaseChatModel
from rapidfuzz import fuzz

from topictrace import settings
from topictrace.rag.documentIngestion.models.graphExtractionModels import EntityResolutionDecision
from topictrace.prompts.ingestion.prompts_for_checking_if_two_names_are_the_same import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)



def group_entities_by_normalized_name(raw_entity_names: list[str]) -> dict[str, list[str]]:
    """This function puts names that look exactly the same (except for capital letters or extra spaces) into groups, so we know they are the same thing."""
    
    grouped_names: dict[str, list[str]] = {}

    for raw_entity_name in raw_entity_names:
        normalized_name = " ".join(raw_entity_name.lower().strip().split())
        grouped_names.setdefault(normalized_name, []).append(raw_entity_name)

    return grouped_names


def find_fuzzy_merge_candidates(
    raw_entity_names: list[str],
    threshold: int = settings.ENTITY_RESOLUTION_FUZZY_THRESHOLD,
) -> list[tuple[str, str]]:
    """This function finds pairs of names that look very similar, like "Apple" and "Apple Inc", and puts them together so we can check if they mean the same thing."""
    
    fuzzy_merge_candidates: list[tuple[str, str]] = []
    
    for left_index, left_name in enumerate(raw_entity_names):
        for right_name in raw_entity_names[left_index + 1:]:
            if fuzz.token_set_ratio(left_name, right_name) >= threshold:
                fuzzy_merge_candidates.append((left_name, right_name))
    return fuzzy_merge_candidates


def split_clear_cases_from_ambiguous_cases(
    pairs_with_scores: list[tuple[str, str, float]],
    high_threshold: float = settings.ENTITY_RESOLUTION_HIGH_THRESHOLD,
    low_threshold: float = settings.ENTITY_RESOLUTION_LOW_THRESHOLD,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """This function checks that we can correctly separate the pairs of names that are super obvious from the pairs that are confusing and need the AI to look at them."""
    same_entity_pairs = []
    ambiguous_pairs = []
    different_entity_pairs = []

    for left, right, score in pairs_with_scores:
        if score >= high_threshold:
            same_entity_pairs.append((left, right, score))
        elif score >= low_threshold:
            ambiguous_pairs.append((left, right, score))
        else:
            different_entity_pairs.append((left, right, score))
            
    return same_entity_pairs, ambiguous_pairs, different_entity_pairs


def build_entity_resolution_messages(ambiguous_pairs: list[tuple[str, str, float]]) -> dict[str, Any]:
    """This makes sure we only send the confusing pairs of names to the AI for review."""
    payload =  {
        "ambiguous_pairs": [
            {
                "left_name": pair[0],
                "right_name": pair[1],
                "similarity_score": pair[2],
            }
            for pair in ambiguous_pairs
        ]
    }

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
    ambiguous_pairs: list[tuple[str, str, float]],
    model: str | None = None,
) -> list[EntityResolutionDecision]:
    """This function asks the AI to look at pairs of names we are confused about, and decide if they are the same thing or different things."""
    if not ambiguous_pairs:
        return []
        
    bind_kwargs = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    if model:
        bind_kwargs["model"] = model
    bound_client = llm_client.bind(**bind_kwargs)
    response = await bound_client.ainvoke(build_entity_resolution_messages(ambiguous_pairs))
    response_payload = json.loads(response.content or "{}")
    
    return [
        EntityResolutionDecision(**decision_row)
        for decision_row in response_payload.get("decisions", [])
    ]
