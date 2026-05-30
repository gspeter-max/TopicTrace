"""Web fetch tool for TopicTrace using Jina Reader."""

import httpx
import asyncio
from topictrace import settings, log
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid, create_cache_key
from langchain_core.tools import tool
from topictrace.session import save_numberd_file


@tool 
async def web_fetch(url, session_path: str) -> list[dict]:
    """Fetch URLs via Jina Reader, save content, return list of result dicts.

    Args:
        url: A single URL string or a list of URLs.
        session_path: Session directory path for caching and file storage.

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
        uncached_urls = []
        for u in urls:
            if not u or not u.strip():
                results.append({
                    "url": u,
                    "status": "error",
                    "content": "URL cannot be empty"
                })
                continue
            cache_key = create_cache_key("fetch", u)
            if is_cache_valid(session_path, cache_key):
                cached = load_from_cache(session_path, cache_key)
                if cached is not None:
                    log.info("cache_hit", url=u)
                    results.append({"url": u, "status": 200, "content": cached})
                    continue
            uncached_urls.append(u)

        if not uncached_urls:
            return results

        # Step 2: Fetch uncached URLs in parallel
        async with httpx.AsyncClient(timeout=settings.FETCH_TIMEOUT_SECONDS) as client:
            tasks = [
                client.get(f"{settings.JINA_READER_BASE_URL}{url}")
                for url in uncached_urls
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Process each response
        for url, response in zip(uncached_urls, responses):
            # Handle network-level exceptions (timeout, connection error, etc.)
            if isinstance(response, Exception):
                log.warning("fetch_failed", url=url, error=str(response))
                results.append({
                    "url": url,
                    "status": "error",
                    "content": f"Request failed: {response}"
                })
                continue

            # Handle non-200 status codes (451, 403, 404, etc.)
            if response.status_code != 200:
                log.warning(
                    f"feteching fail due to : {response.status_code}",
                    url=url,
                    status_code=response.status_code,
                )
                results.append({
                    "url": url,
                    "status": response.status_code,
                    "content": None,
                })
                continue

            # Success: save to file and cache
            content = response.text
            save_numberd_file(
                content=f"<!-- Source: {url} -->\n\n{content}",
                subdir="fetched_pages",
                prefix="page",
                session_path=session_path
            )
            cache_key = create_cache_key("fetch", url)
            save_to_cache(session_path, cache_key, content)
            log.info("fetch_success", url=url, chars=len(content))
            results.append({"url": url, "status": 200, "content": content})

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
