"""LLM provider for TopicTrace using OpenAI-compatible API."""

from typing import Literal
from topictrace import settings
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
import httpx

DEFAULT_MODEL = settings.LLM_CONFIG.MISTRAL_AI.LLM_MODEL

async def build_mistral_client(api_key: str | None = None) -> AsyncOpenAI:
    config = settings.LLM_CONFIG.MISTRAL_AI
    resolved_key = api_key or config.LLM_API_KEY
    if not resolved_key:
        raise EnvironmentError("MISTRAL_API_KEY is not configured")
    return AsyncOpenAI(
        api_key=resolved_key,
        base_url=config.LLM_BASE_URL,
    )


async def build_deepseek_client(api_key: str | None = None) -> AsyncOpenAI:
    config = settings.LLM_CONFIG.DEEPSEEK_AI
    resolved_key = api_key or config.LLM_API_KEY
    if not resolved_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not configured")
    return AsyncOpenAI(
        api_key=resolved_key,
        base_url=config.LLM_BASE_URL,
    )


_common_headers = {"Accept-Encoding": "identity"}  # no gzip — gateway sends broken compressed responses


def get_llm(provider: Literal["DEEPSEEK_AI", "MISTRAL_AI"] = 'DEEPSEEK_AI'):
    http_client = httpx.Client(headers=_common_headers, timeout=settings.LLM_CLIENT_TIMEOUT_SECONDS)
    http_async_client = httpx.AsyncClient(headers=_common_headers, timeout=settings.LLM_CLIENT_TIMEOUT_SECONDS)
    
    config = getattr(settings.LLM_CONFIG, provider)
        
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        temperature=settings.SUMMARIZE_TEMPERATURE,
        http_client=http_client,
        http_async_client=http_async_client,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "high"
            }
        } if provider == "DEEPSEEK_AI" else None
    )


def get_llm_with_tools(tools : list ):
    llm = get_llm() 
    return llm.bind_tools(tools) 
