from typing import LiteralString

from neo4j import AsyncGraphDatabase


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def execute_query(
        self, query: LiteralString, parameters: dict | None = None
    ) -> list[dict]:
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()
