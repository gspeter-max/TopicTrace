
from openai import OpenAI
from topictrace import settings


def call_llm(messages: list[dict], temperature: float = None, max_tokens: int = None) -> str:
    """
    Call GLM-5.1 via NVIDIA NIM and return the response.

    """
    if not settings.NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY not found. "
            "Set it in your .env file: NVIDIA_API_KEY=your-key-here"
        )

    client = OpenAI(
        base_url=settings.NVIDIA_BASE_URL,
        api_key=settings.NVIDIA_API_KEY
    )

    completion = client.chat.completions.create(
        model=settings.NVIDIA_MODEL,
        messages=messages,
        temperature=temperature or settings.SUMMARIZE_TEMPERATURE,
        max_tokens=max_tokens or settings.SUMMARIZE_MAX_TOKENS,
        stream=True
    )

    response_parts = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None) is not None:
            response_parts.append(delta.content)

    return "".join(response_parts)
