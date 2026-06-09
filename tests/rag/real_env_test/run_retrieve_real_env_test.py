"""
Real environment smoke test for the Retrieval pipeline.

This script runs actual queries against the live local Neo4j database using
the new LangGraph retrieval pipeline. It requires that the database already
contains ingested data (e.g., from running run_real_env_test.py on a PDF).

Usage:
    PYTHONPATH=src python tests/real_env_test/run_retrieve_real_env_test.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import structlog

# Add src to path if not running with PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from topictrace.rag.documentRetrieve.retrieve import handle_query
from topictrace.server.schemas.rag.retrieveModels import QueryRequest

# Configure structured logging for the test output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger()


# Test cases covering different paths of the LangGraph
TEST_QUERIES = [
    {
        "name": "Simple Factoid (Vector Fast Path)",
        "query": "What is the email address of Pankaj Kumar?",
        "expected_intent": "simple",
        # Should ideally be found via vector search alone, without graph
    },
    {
        "name": "Simple Complex (Graph Escalation or Complex Intent)",
        "query": "Which companies has Pankaj Kumar worked for, and what were his roles?",
        "expected_intent": "complex",
        # Or it might route simple -> fail grader -> escalate. Either way, graph should be used.
    },
    {
        "name": "Skill Verification",
        "query": "Does Pankaj Kumar have experience with Python and FastAPI?",
        "expected_intent": "simple",
    },
]


async def run_query(query_def: dict) -> dict:
    """Run a single query and return the result data."""
    log.info(f"--- Running Query: {query_def['name']} ---", query=query_def["query"])

    req = QueryRequest(query=query_def["query"], top_k=5, top_k_rerank=3)

    # Run the pipeline
    start_time = asyncio.get_event_loop().time()
    res = await handle_query(req)
    end_time = asyncio.get_event_loop().time()

    duration = end_time - start_time

    # Log results
    log.info(
        "Query complete",
        duration=f"{duration:.2f}s",
        intent=res.intent,
        used_graph=res.used_graph_search,
        reason_for_graph=res.reason_for_graph_search,
    )

    print("\n--- ANSWER ---")
    print(res.answer)
    print("--------------\n")

    return {
        "name": query_def["name"],
        "query": query_def["query"],
        "expected_intent": query_def["expected_intent"],
        "actual_intent": res.intent,
        "used_graph_search": res.used_graph_search,
        "reason_for_graph_search": res.reason_for_graph_search,
        "answer": res.answer,
        "duration_seconds": duration,
        "num_context_items_used": len(res.context_used),
    }


async def main():
    log.info("Starting Retrieval Real Environment Test")

    results = []
    success = True

    for q in TEST_QUERIES:
        try:
            res_data = await run_query(q)
            results.append(res_data)
        except Exception as e:
            log.error("Query failed with exception", query=q["query"], error=str(e))
            results.append({"name": q["name"], "query": q["query"], "error": str(e)})
            success = False

    # Save report
    report_dir = os.path.join(os.path.dirname(__file__), "real_env_test_results")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"retrieve_test_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(
        "Test suite complete",
        report_path=report_path,
        status="success" if success else "failed",
    )


if __name__ == "__main__":
    asyncio.run(main())
