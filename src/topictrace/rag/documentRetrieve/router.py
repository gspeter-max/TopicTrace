"""
LLM-based query intent router.

Takes a query string and uses Mistral's structured output (json_object)
to classify the intent as either "simple" or "complex".
No keywords or regex, pure LLM routing.
Defaults to "simple" on failure.
"""
import json
from topictrace import log
from typing import Literal

from topictrace.provider.llm import build_mistral_client, DEFAULT_MODEL
from topictrace.prompts.router_intent_classifier import ROUTER_PROMPT




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
        client = await build_mistral_client()

        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content
        if not content:
            log.error("Router received empty response from LLM, defaulting to 'simple'")
            return "simple"

        data = json.loads(content)
        intent = data.get("intent", "").lower()

        if intent in ("simple", "complex"):
            return intent  # type: ignore

        log.warning("Router received unexpected intent, defaulting to 'simple'", parsed_intent=intent)
        return "simple"

    except Exception as e:
        log.warning("Router LLM call failed, defaulting to 'simple'", error=str(e))
        return "simple"
