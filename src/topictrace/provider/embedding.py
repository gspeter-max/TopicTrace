import asyncio
import json
from typing import overload

import requests

from topictrace import log, settings


class embeddingModel:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        embeddingModel: str | None = None,
        max_concurrency: int | None = None,
    ):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.embedModel = (
            embeddingModel or settings.EMBEDDING_CONFIG.JINA_EMBEDDING_MODEL
        )
        self.url = base_url or settings.EMBEDDING_CONFIG.JINA_BASE_URL
        self.max_concurrency = (
            max_concurrency or settings.EMBEDDING_CONFIG.MAX_CONCURRENCY
        )
        self.task = settings.EMBEDDING_CONFIG.JINA_EMBEDDING_TASK

    @overload
    async def generateEmebedding(self, texts: str) -> list[float]: ...

    @overload
    async def generateEmebedding(self, texts: list[str]) -> list[list[float]]: ...

    async def generateEmebedding(
        self, texts: str | list[str]
    ) -> list[float] | list[list[float]]:
        is_string = isinstance(texts, str)
        if is_string:
            texts = [texts]

        sem = asyncio.Semaphore(self.max_concurrency)
        async with sem:
            data = {"model": self.embedModel, "task": self.task, "input": texts}

            response = requests.post(
                self.url, headers=self.headers, data=json.dumps(data)
            )
            if response.status_code != 200:
                log.error(
                    "Jina API error",
                    status_code=response.status_code,
                    response_text=response.text,
                )
                raise Exception(f"Error generating embedding: {response.status_code}")

            embeddings = [
                object_embedding["embedding"]
                for object_embedding in response.json()["data"]
            ]
            return embeddings[0] if is_string else embeddings
