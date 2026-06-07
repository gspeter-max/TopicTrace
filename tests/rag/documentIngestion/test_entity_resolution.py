from topictrace.rag.documentIngestion.entityResolution import (
    group_entities_by_normalized_name,
    find_fuzzy_merge_candidates,
    split_clear_cases_from_ambiguous_cases,
)


def test_group_entities_by_normalized_name_merges_case_only_differences():
    """This test makes sure that when we have the same name written with different capital letters, we group them as the exact same name."""
    grouped = group_entities_by_normalized_name(["Tesla", "tesla", "TESLA"])
    assert len(grouped) == 1


def test_find_fuzzy_merge_candidates_detects_company_suffix_variants():
    """This test makes sure we can find and group names that look very similar, like a company name with or without the word 'Inc'."""
    candidates = find_fuzzy_merge_candidates(["Apple", "Apple Inc", "Microsoft"])
    assert ("Apple", "Apple Inc") in candidates or ("Apple Inc", "Apple") in candidates


def test_split_clear_cases_from_ambiguous_cases_keeps_mid_band_for_later_resolution():
    """This test checks that we can correctly separate the pairs of names that are super obvious from the pairs that are confusing and need the AI to look at them."""
    same_entity_pairs, ambiguous_pairs, different_entity_pairs = split_clear_cases_from_ambiguous_cases(
        [
            ("Elon Musk", "Elon Musk", 0.99),
            ("Elon Musk", "Mr Musk", 0.72),
            ("Elon Musk", "Apple", 0.11),
        ]
    )
    assert ambiguous_pairs == [("Elon Musk", "Mr Musk", 0.72)]

def test_find_fuzzy_merge_candidates_with_empty_input():
    """Edge case: ensure empty list doesn't crash."""
    candidates = find_fuzzy_merge_candidates([])
    assert len(candidates) == 0

def test_split_clear_cases_from_ambiguous_cases_with_extreme_values():
    """Edge case: test exact 1.0 and 0.0 scores."""
    same, ambiguous, diff = split_clear_cases_from_ambiguous_cases([
        ("A", "A", 1.0),
        ("A", "B", 0.0)
    ])
    assert len(same) == 1
    assert len(diff) == 1
    assert len(ambiguous) == 0

from topictrace.rag.documentIngestion.entityResolution import build_entity_resolution_review_payload

def test_build_entity_resolution_review_payload_only_contains_ambiguous_cases():
    """This test makes sure we only send the confusing pairs of names to the AI for review, and not the obvious ones."""
    payload = build_entity_resolution_review_payload(
        ambiguous_pairs=[
            ("Elon", "Mr Musk", 0.72),
            ("Board", "Board of Directors", 0.69),
        ]
    )

    assert len(payload["ambiguous_pairs"]) == 2
    assert payload["ambiguous_pairs"][0]["left_name"] == "Elon"

def test_build_entity_resolution_review_payload_handles_empty():
    """Edge case test: handle empty ambiguous pairs without crashing."""
    payload = build_entity_resolution_review_payload([])
    assert len(payload["ambiguous_pairs"]) == 0

