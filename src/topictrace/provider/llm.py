"""LLM provider for TopicTrace using OpenAI-compatible API."""

from topictrace import settings
from langchain_openai import ChatOpenAI
import httpx 

def get_llm():
    http_client = httpx.Client(
        headers={"Accept-Encoding": "identity"},  # accept plain text ( no transformation of data )
        timeout=60
    )
    return ChatOpenAI(
        base_url = settings.LLM_BASE_URL,
        model = settings.LLM_MODEL,
        api_key = settings.LLM_API_KEY,
        temperature=settings.SUMMARIZE_TEMPERATURE,
        http_client=http_client
    )

def get_llm_with_tools(tools : list ):
    llm = get_llm() 
    return llm.bind_tools(tools) 


