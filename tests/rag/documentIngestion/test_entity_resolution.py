from topictrace.rag.documentIngestion.entityResolution import (
    find_fuzzy_merge_candidates,
    split_clear_cases_from_ambiguous_cases,
    build_entity_resolution_messages
)


def test_find_fuzzy_merge_candidates_detects_company_suffix_variants():
    """This test makes sure we can find and group names that look very similar, like a company name with or without the word 'Inc'."""
    candidates, left_candidates = find_fuzzy_merge_candidates({"Apple", "Apple Inc", "Microsoft"})
    assert ("Apple", "Apple Inc") in candidates or ("Apple Inc", "Apple") in candidates
    assert "Microsoft" in left_candidates


def test_split_clear_cases_from_ambiguous_cases_separates_obvious_from_different():
    """This test checks that we can correctly separate the pairs of names that are super obvious from the different ones."""
    same, different = split_clear_cases_from_ambiguous_cases(
        [
            ("Elon Musk", "Elon Musk", 0.99),
            ("Elon Musk", "Mr Musk", 0.72),
            ("Elon Musk", "Apple", 0.11),
        ]
    )
    assert same == [("Elon Musk", "Elon Musk", 0.99)]
    assert "Mr Musk" in different
    assert "Apple" in different


def test_find_fuzzy_merge_candidates_with_empty_input():
    """Edge case: ensure empty set doesn't crash."""
    candidates, left = find_fuzzy_merge_candidates(set())
    assert len(candidates) == 0
    assert len(left) == 0


def test_split_clear_cases_from_ambiguous_cases_with_extreme_values():
    """Edge case: test exact 1.0 and 0.0 scores."""
    same, different = split_clear_cases_from_ambiguous_cases([
        ("A", "A", 1.0),
        ("A", "B", 0.0)
    ])
    assert len(same) == 1
    assert "B" in different


def test_build_entity_resolution_messages_with_flat_list():
    """This test makes sure we send the flat list of entity names to the AI for review."""
    message = build_entity_resolution_messages({"Elon", "Board"})
    assert message[0]["role"] == "system"
    assert message[1]["role"] == "user"
    assert "Elon" in message[1]["content"] 


def test_build_entity_resolution_review_payload_handles_empty():
    """Edge case test: handle empty set without crashing."""
    import json 

    message = build_entity_resolution_messages(set())
    user_content = message[1]["content"] 
    json_start = user_content.find("{")
    json_end = user_content.rfind("}")

    json_content = json.loads(user_content[json_start:json_end + 1])
    assert len(json_content["entity_names"]) == 0
