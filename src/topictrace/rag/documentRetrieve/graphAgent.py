"""
Graph Agent.

Explores the Neo4j knowledge graph starting from a set of extracted entity_ids.
Uses a 1-hop traversal first. If no connections are found, it expands to 2-hop.
Formats the gathered relationships into a readable fact sheet for the LLM.
"""
from typing import Any
from topictrace import log

from topictrace.db.neo4j import Neo4jClient
from topictrace.db.neo4j.cypherQuerys import fetch_entity_neighbors_1hop





def _rows_to_fact_sheet(rows: list[dict[str, Any]]) -> str:
    """Converts raw Cypher rows into a human-readable text block."""
    if not rows:
        return ""

    lines = []
    for r in rows:
        source = r.get("source", "Unknown")
        rel = r.get("rel_type", "RELATES_TO")
        target = r.get("target", "Unknown")
        evidence = r.get("evidence_text", "").replace("\n", " ").strip()

        line = f"- {source} {rel} {target}"
        if evidence:
            line += f" (Evidence: '{evidence}')"
        lines.append(line)

    return "\n".join(lines)


async def gather_graph_facts(client: Neo4jClient, entity_ids: list[str]) -> str:
    """
    Given seed entity_ids, crawls the graph and returns a formatted text of facts.
    Uses 1-hop traversal to gather direct relationships.
    """
    if not entity_ids:
        return ""

    log.debug("Graph agent starting 1-hop traversal", entity_count=len(entity_ids))
    rows = await fetch_entity_neighbors_1hop(client, entity_ids)

    if not rows:
        log.debug("1-hop empty, returning no facts")
        return ""

    log.debug("Graph agent retrieved facts", fact_count=len(rows))
    return "GRAPH KNOWLEDGE:\n" + _rows_to_fact_sheet(rows)
