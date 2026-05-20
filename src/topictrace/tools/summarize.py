"""
Summarize tool for TopicTrace using GLM-5.1 via NVIDIA NIM.

Uses the call_llm function from provider/llm.py.
GLM-5.1 is a powerful model for summarization tasks.
"""

import os
from topictrace import settings
from topictrace.provider.llm import call_llm


def _save_summary_to_file(summary: str, session_path: str) -> None:
    """
    Save summary to the summaries directory.

    Args:
        summary: The summary text to save
        session_path: Path to the session directory
    """
    summaries_dir = os.path.join(session_path, "summaries")

    existing_files = [f for f in os.listdir(summaries_dir) if f.endswith(".md")]
    next_number = len(existing_files) + 1

    filename = f"summary_{next_number}.md"
    filepath = os.path.join(summaries_dir, filename)

    with open(filepath, "w") as f:
        f.write(summary)


def summarize(content: str, query: str, session_path: str) -> str:
    """
    Summarize content using GLM-5.1 based on the user's query.

    Args:
        content: The full text to summarize (from web_fetch)
        query: The user's original question (for context)
        session_path: Path to the session directory for saving

    Returns:
        A concise summary string

    Raises:
        ValueError: If content is empty
        Exception: If NVIDIA NIM API call fails
    """
    if not content or not content.strip():
        raise ValueError("Content cannot be empty for summarization")

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

    summary = call_llm(messages)

    # Save summary to file
    _save_summary_to_file(summary, session_path)

    return summary
