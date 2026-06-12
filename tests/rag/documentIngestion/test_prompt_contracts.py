from topictrace.prompts import get_system_prompt, get_user_prompt


def test_extraction_prompt_mentions_related_to_as_fallback_and_evidence_text():
    prompt = get_system_prompt("extraction", {"schema_text": "MEMBER_OF, USES"})

    assert "RELATED_TO" in prompt
    assert "Never invent a new relationship type" in prompt
    assert "evidence_text" in prompt


def test_resolution_prompt_requires_exact_json_shape():
    prompt = get_system_prompt("resolution")

    assert "decisions" in prompt
    assert "canonical_name" in prompt


def test_extraction_system_prompt_defines_explicit_json_keys():
    prompt = get_system_prompt("extraction", {"schema_text": "MEMBER_OF, USES"})

    # Check for entity keys
    assert "entity_name" in prompt
    assert "entity_type" in prompt
    assert "chunk_id" in prompt

    # Check for relationship keys
    assert "source_entity_name" in prompt
    assert "target_entity_name" in prompt
    assert "relationship_type" in prompt


def test_extraction_user_prompt_contains_one_shot_example():
    prompt = get_user_prompt(
        "extraction",
        {
            "chunk_id": "doc1::0",
            "chunk_text": "Alice engineers software using Python at Neo4j.",
        },
    )

    assert "Example Output Format" in prompt or "Example" in prompt
    assert "Alice" in prompt  # Basic check that example content exists
    assert "entity_name" in prompt  # Check that the example uses the correct keys
