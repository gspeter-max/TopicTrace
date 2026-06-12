"""
LLM Grader.

Analyzes retrieved chunks for a specific query and determines if they contain
sufficient information to answer the query. If insufficient, provides a reason
to justify graph escalation.
"""

import json
from typing import cast

from pydantic import BaseModel, Field

from topictrace import log
from topictrace.prompts import get_system_prompt
from topictrace.provider.llm import get_llm


class GraderResult(BaseModel):
    sufficient: bool = Field(
        ...,
        description="True if the chunks contain enough information to fully answer the query.",
    )
    reason: str = Field(
        ...,
        description="If sufficient is False, explain exactly what information is missing. If True, this can be empty.",
    )
    answer: str = Field(
        "",
        description="If sufficient is True, generate the final answer to the user's query here.",
    )


async def grade_chunks(query: str, chunks: list[str]) -> GraderResult:
    """
    Grades whether the provided chunks are sufficient to answer the query, and attempts to generate the answer.
    If JSON parsing fails, defaults to sufficient=False to err on the side of safety (escalation).
    """
    if not chunks:
        return GraderResult(sufficient=False, reason="No chunks provided.", answer="")

    try:
        llm = get_llm("MISTRAL_AI")
        bound_llm = llm.bind(response_format={"type": "json_object"}, temperature=0.0)

        # Combine chunks into a single text block
        chunks_text = "\n\n---\n\n".join(chunks)

        user_content = f"QUERY: {query}\n\nRETRIEVED DOCUMENTS:\n{chunks_text}"

        response = await bound_llm.ainvoke(
            [
                {"role": "system", "content": get_system_prompt("grader")},
                {"role": "user", "content": user_content},
            ]
        )

        content = response.content
        if not content:
            log.warning(
                "Grader received empty response from LLM, defaulting to insufficient"
            )
            return GraderResult(
                sufficient=False, reason="Empty LLM response.", answer=""
            )
        
        data = json.loads(cast(str, content))      
        result = GraderResult(
            sufficient=bool(data.get("sufficient", False)),
            reason=str(data.get("reason", "No reason provided by LLM.")),
            answer=str(data.get("answer", "")),
        )

        if not result.sufficient:
            log.info("Grader determined chunks are insufficient", reason=result.reason)
        else:
            log.info("Grader generated answer from chunks")

        return result

    except json.JSONDecodeError as e:
        log.warning(
            "Grader failed to parse JSON from LLM, defaulting to insufficient",
            error=str(e),
        )
        return GraderResult(
            sufficient=False, reason=f"Grader JSON parse error: {e}", answer=""
        )
    except Exception as e:
        log.error("Grader LLM call failed, defaulting to insufficient", error=str(e))
        return GraderResult(
            sufficient=False, reason=f"Grader LLM error: {e}", answer=""
        )
