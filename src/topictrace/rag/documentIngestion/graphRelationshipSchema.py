ALLOWED_RELATIONSHIP_TYPES = [
    "RELATED_TO",
    "PART_OF",
    "DEPENDS_ON",
    "ASSIGNED_TO",
    "BLOCKED_BY",
    "OWNS",
    "REPORTS_TO",
    "LOCATED_IN",
]


def get_relationship_schema_prompt_text() -> str:
    """This gives us the exact list of allowed connections as text, so we can tell the AI exactly what words to use when connecting things."""
    return "\n".join(
        f"- {relationship_type}" for relationship_type in ALLOWED_RELATIONSHIP_TYPES
    )
