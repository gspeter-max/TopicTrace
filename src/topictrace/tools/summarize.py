"""
Summarize tool for TopicTrace using GLM-5.1 via NVIDIA NIM.

Uses the OpenAI client to connect to NVIDIA's NIM API endpoint.
GLM-5.1 is a powerful model for summarization tasks.
"""

import os
from openai import OpenAI
from topictrace import settings


def _save_summary_to_file(summary: str, session_path: str) -> None:
    """
    Save summary to the summaries directory.

    Creates a numbered file like summary_1.md, summary_2.md, etc.

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

    Sends the content and query to GLM-5.1 via NVIDIA NIM.
    The model produces a concise summary relevant to the query.
    Results are streamed and saved to session folder.

    Args:
        content: The full text to summarize (from web_fetch)
        query: The user's original question (for context)
        session_path: Path to the session directory for saving

    Returns:
        A concise summary string

    Raises:
        ValueError: If content is empty or API key is missing
        Exception: If NVIDIA NIM API call fails
    """
    if not content or not content.strip():
        raise ValueError("Content cannot be empty for summarization")

    if not settings.NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY not found. "
            "Set it in your .env file: NVIDIA_API_KEY=your-key-here"
        )

    client = OpenAI(
        base_url=settings.NVIDIA_BASE_URL,
        api_key=settings.NVIDIA_API_KEY
    )

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

    # Call GLM-5.1 via NVIDIA NIM with streaming
    completion = client.chat.completions.create(
        model=settings.NVIDIA_MODEL,
        messages=messages,
        temperature=settings.SUMMARIZE_TEMPERATURE,
        max_tokens=settings.SUMMARIZE_MAX_TOKENS,
        stream=True
    )

    # Collect streamed response
    summary_parts = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None) is not None:
            summary_parts.append(delta.content)

    summary = "".join(summary_parts)

    # Save summary to file
    _save_summary_to_file(summary, session_path)

    return summary
