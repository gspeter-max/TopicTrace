import json
from typing import Any

from langchain_core.language_models import BaseChatModel

from topictrace.prompts.ingestion.extraction import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from topictrace.rag.documentIngestion.graphRelationshipSchema import (
    get_relationship_schema_prompt_text,
)
from topictrace.rag.documentIngestion.models.graphExtractionModels import (
    ChunkGraphExtractionResult,
)


def build_graph_extraction_messages(chunk: dict[str, Any]) -> list[dict[str, str]]:
    """This function puts together the instructions and the text piece so we can ask the AI to find names and connections."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                schema_text=get_relationship_schema_prompt_text()
            ),
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                chunk_id=chunk["chunk_id"], chunk_text=chunk["text"]
            ),
        },
    ]


async def extract_graph_data_from_chunk(
    *,
    llm_client: BaseChatModel,
    chunk: dict[str, Any],
    model: str | None = None,
) -> ChunkGraphExtractionResult:
    """This function looks at a small piece of the text and asks the AI to find all the important names
    (like people or places) and how they connect to each other, like drawing lines between dots."""
    bind_kwargs = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    if model:
        bind_kwargs["model"] = model
    bound_client = llm_client.bind(**bind_kwargs)
    response = await bound_client.ainvoke(build_graph_extraction_messages(chunk))
    response_payload = json.loads(response.content or "{}")
    return ChunkGraphExtractionResult(**response_payload)
