"""Summarize tool for TopicTrace using GLM-5.1 via NVIDIA NIM."""

from topictrace import settings, log
from topictrace.provider.llm import get_llm
from langchain_core.tools import tool 
from topictrace.session import save_numberd_file

@tool
async def summarize(content: str, query: str) -> str:
    """Summarize content using llm based on the query and save it to the session folder."""
    from topictrace.session import create_session
    session_path = create_session(query[:50])

    if not content or not content.strip():
        log.warning("Content cannot be empty for summarization")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. "
                "Summarize the provided content in relation to the user's query. "
                "Be concise, factual, and focus on exam-relevant information. "
                "Output only the summary, no preamble."
            )
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Content to summarize:\n{content[:settings.SUMMARIZE_MAX_INPUT_CHARS]}"
            )
        }
    ]
    
    llm  = get_llm()
    response = await llm.ainvoke(messages)
    save_numberd_file(
        content=response.content,
        subdir="summaries",
        prefix="summary",
        session_path=session_path
    )
    return response.content
