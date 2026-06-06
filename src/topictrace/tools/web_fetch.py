"""Web fetch tool for TopicTrace using Jina Reader + LLM summarization."""

import asyncio
import httpx
from langchain_core.tools import tool
from topictrace import settings, log
from topictrace.tools.cache import save_to_cache, load_from_cache, generate_fetch_cache_key
from topictrace.provider.llm import get_llm


@tool
async def web_fetch(url, query: str) -> list[dict]:
    """Fetch URLs via Jina Reader, summarize with LLM, cache results.

    Args: 
        url: A single URL string or a list of URLs.
        query: The original research query (used for cache key and summarization).

    Returns:
        List of dicts: [{"url": ..., "status": 200, "content": "..."}, ...]
    """

    # Normalize input: accept str or list
    if isinstance(url, str):
        urls = [url]
    elif isinstance(url, list):
        urls = url
    else:
        error_message = (
            f"url parameter must be str or list, got {type(url).__name__}"
        )
        log.warning("invalid_input", error=error_message)
        return [{"url": "", "status": "error", "content": error_message}]

    results = []

    try:
        # Step 1: Separate cached vs uncached URLs
        uncached_urls = []  # list of (url, cache_key) tuples
        for u in urls:
            if not u or not u.strip():
                results.append({
                    "url": u,
                    "status": "error",
                    "content": "URL cannot be empty"
                })
                continue

            cache_key = generate_fetch_cache_key(query, u)
            cached = load_from_cache(cache_key)
            if cached is not None:
                log.info("[WEB_FETCH] cache hit", url=u)
                results.append({"url": u, "status": 200, "content": cached})
                continue
            uncached_urls.append((u, cache_key))

        if not uncached_urls:
            return results

        # Step 2: Fetch uncached URLs in parallel
        async with httpx.AsyncClient(timeout=settings.FETCH_TIMEOUT_SECONDS) as client:
            tasks = [
                client.get(f"{settings.JINA_READER_BASE_URL}{u}")
                for u, _ in uncached_urls
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Process each response
        for (u, cache_key), http_response in zip(uncached_urls, responses):
            # Handle network-level exceptions (timeout, connection error, etc.)
            if isinstance(http_response, Exception):
                log.warning("fetch_failed", url=u, error=str(http_response))
                results.append({
                    "url": u,
                    "status": "error",
                    "content": f"Request failed: {http_response}"
                })
                continue

            # Handle non-200 status codes (451, 403, 404, etc.)
            if http_response.status_code != 200:
                log.warning(
                    f"fetch failed: {http_response.status_code}",
                    url=u,
                    status_code=http_response.status_code,
                )
                results.append({
                    "url": u,
                    "status": http_response.status_code,
                    "content": None,
                })
                continue
            content = http_response.text

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
            llm = get_llm()
            llm_response = await llm.ainvoke(messages)

            save_to_cache(cache_key, llm_response.content)

            results.append({"url": u, "status": 200, "content": llm_response.content})

    except httpx.HTTPError as e:
        log.error("http_error", error=str(e))
        results.append({"url": "", "status": "error", "content": f"HTTP error: {e}"})
    except OSError as e:
        log.error("os_error", error=str(e))
        results.append({"url": "", "status": "error", "content": f"File system error: {e}"})
    except Exception as e:
        log.error("unexpected_error", error=str(e))
        results.append({"url": "", "status": "error", "content": f"Unexpected error: {e}"})
    finally:
        log.info("fetch_batch_complete", total=len(urls), success=sum(1 for r in results if r["status"] == 200))

    return results
