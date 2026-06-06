
from openai import AsyncOpenAI
from config import mistral_api_key


async def build_mistral_client(
    api_key: str | None = None,
    base_url: str = MISTRAL_BASE_URL,
) -> AsyncOpenAI:
    resolved_api_key = (api_key or mistral_api_key or "").strip()
    if not resolved_api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. "
            "Set it in your .env file, expose it from src.config, or pass it explicitly."
        )

    return AsyncOpenAI(api_key=resolved_api_key, base_url=base_url)
