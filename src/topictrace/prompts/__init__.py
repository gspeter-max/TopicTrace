from typing import Any, Callable, Literal

from topictrace import log
from topictrace.prompts.answer_generator import build_final_answer_prompt
from topictrace.prompts.grader_chunk_evaluator import GRADER_PROMPT
from topictrace.prompts.ingestion.extraction import (
    SYSTEM_PROMPT as EXTRACTION_SYSTEM,
)
from topictrace.prompts.ingestion.extraction import (
    USER_PROMPT_TEMPLATE as EXTRACTION_USER,
)
from topictrace.prompts.ingestion.resolution import (
    SYSTEM_PROMPT as RESOLUTION_SYSTEM,
)
from topictrace.prompts.ingestion.resolution import (
    USER_PROMPT_TEMPLATE as RESOLUTION_USER,
)
from topictrace.prompts.research_agent import (
    get_system_prompt as get_research_system_prompt,
)
from topictrace.prompts.research_agent import (
    get_user_prompt as get_research_user_prompt,
)
from topictrace.prompts.router_intent_classifier import ROUTER_PROMPT

PromptType = Literal[
    "router",
    "grader",
    "answer_generator",
    "research",
    "extraction",
    "resolution",
    "compact",
]


def _require(v: dict[str, Any], key: str) -> Any:
    if key not in v:
        raise ValueError(f"'{key}' required in input_vars.")
    return v[key]


PROMPT_BUILDERS: dict[PromptType, Callable[[dict[str, Any]], str]] = {
    "router": lambda _: ROUTER_PROMPT,
    "grader": lambda _: GRADER_PROMPT,
    "answer_generator": lambda v: build_final_answer_prompt(
        _require(v, "context_block")
    ),
    "research": lambda _: get_research_system_prompt(),
    "extraction": lambda v: EXTRACTION_SYSTEM.format(
        schema_text=_require(v, "schema_text")
    ),
    "resolution": lambda _: RESOLUTION_SYSTEM,
    "compact": lambda _: (
        "You are an assistant tasked to summarize and extract key memory/information from the conversation."
    ),
}

USER_PROMPT_BUILDERS: dict[PromptType, Callable[[dict[str, Any]], str]] = {
    "research": lambda v: get_research_user_prompt(
        query=_require(v, "query"), depth=v.get("depth", "standard")
    ),
    "extraction": lambda v: EXTRACTION_USER.format(
        chunk_id=_require(v, "chunk_id"), chunk_text=_require(v, "chunk_text")
    ),
    "resolution": lambda v: RESOLUTION_USER.format(
        payload_json=_require(v, "payload_json")
    ),
    "compact": lambda v: (
        f"Summarize the following conversation/response to extract key information to remember: {_require(v, 'response_text')}"
    ),
}


def get_system_prompt(prompt_type: str, vars_dict: dict[str, Any] | None = None) -> str:
    """
    Retrieves the system prompt for the specified prompt type, applying dynamic formatting
    using values provided in the `vars_dict` dictionary.
    Args:
        prompt_type: The identifier for the desired prompt.
        vars_dict: A dictionary of key-value pairs used to format the prompt.
    Returns:
        The formatted system prompt as a string.
    Raises:
        ValueError: If a required variable is missing in `vars_dict` or if the `prompt_type` is unknown.
    """
    v = vars_dict or {}
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown system prompt type: {prompt_type}")
    return PROMPT_BUILDERS[prompt_type](v)


def get_user_prompt(prompt_type: str, vars_dict: dict[str, Any] | None = None) -> str:
    """
    Retrieves the user prompt for the specified prompt type, applying dynamic formatting
    using values provided in the `vars_dict` dictionary.
    Args:
        prompt_type: The identifier for the desired prompt.
        vars_dict: A dictionary of key-value pairs used to format the prompt.
    Returns:
        The formatted user prompt as a string.
    Raises:
        ValueError: If a required variable is missing in `vars_dict`, if the prompt type does
                    not support a user prompt, or if the `prompt_type` is unknown.
    """
    v = vars_dict or {}
    if prompt_type in ("router", "grader", "answer_generator"):
        raise ValueError(f"Prompt type '{prompt_type}' does not support user prompts.")
    if prompt_type not in USER_PROMPT_BUILDERS:
        raise ValueError(f"Unknown user prompt type: {prompt_type}")

    return USER_PROMPT_BUILDERS[prompt_type](v)
