"""
LLM-based query intent router.

Takes a query string and uses Mistral's structured output (json_object)
to classify the intent as either "simple" or "complex".
No keywords or regex, pure LLM routing.
Defaults to "simple" on failure.
"""

import json
from typing import Literal

from topictrace import log
from topictrace.prompts import get_system_prompt
from topictrace.provider.llm import get_llm

IntentType = Literal["simple", "complex"]


async def classify_intent(query: str) -> IntentType:
    """
    Classifies a user query's intent as "simple" or "complex" using an LLM.

    Args:
        query: The user's query string.

    Returns:
        "simple" or "complex". Defaults to "simple" if parsing fails or LLM gives unexpected output.
    """
    try:
        llm = get_llm("MISTRAL_AI")
        bound_llm = llm.bind(response_format={"type": "json_object"}, temperature=0.0)

        response = await bound_llm.ainvoke(
            [
                {"role": "system", "content": get_system_prompt("router")},
                {"role": "user", "content": query},
            ]
        )

        content = response.content
        if not content:
            log.error("Router received empty response from LLM, defaulting to 'simple'")
            return "simple"

        content_str = content if isinstance(content, str) else json.dumps(content)
        data = json.loads(content_str)
        intent = data.get("intent", "").lower()

        if intent in ("simple", "complex"):
            return intent

        log.warning(
            "Router received unexpected intent, defaulting to 'simple'",
            parsed_intent=intent,
        )
        return "simple"

    except Exception as e:
        log.warning("Router LLM call failed, defaulting to 'simple'", error=str(e))
        return "simple"
