from topictrace.prompts.ingestion.extraction import (
    SYSTEM_PROMPT as EXTRACTION_SYSTEM,
    USER_PROMPT_TEMPLATE as EXTRACTION_USER
)
from topictrace.prompts.ingestion.resolution import (
    SYSTEM_PROMPT as RESOLUTION_SYSTEM
)


def test_extraction_prompt_mentions_related_to_as_fallback_and_evidence_text():
    prompt = EXTRACTION_SYSTEM

    assert 'RELATED_TO' in prompt
    assert 'Never invent a new relationship type' in prompt
    assert 'evidence_text' in prompt


def test_resolution_prompt_requires_exact_json_shape():
    prompt = RESOLUTION_SYSTEM

    assert "decisions" in prompt
    assert "canonical_name" in prompt


def test_extraction_system_prompt_defines_explicit_json_keys():
    prompt = EXTRACTION_SYSTEM
    
    # Check for entity keys
    assert "entity_name" in prompt
    assert "entity_type" in prompt
    assert "chunk_id" in prompt
    
    # Check for relationship keys
    assert "source_entity_name" in prompt
    assert "target_entity_name" in prompt
    assert "relationship_type" in prompt


def test_extraction_user_prompt_contains_one_shot_example():
    prompt = EXTRACTION_USER
    
    assert "Example Output Format" in prompt or "Example" in prompt
    assert "Alice" in prompt # Basic check that example content exists
    assert "entity_name" in prompt # Check that the example uses the correct keys
