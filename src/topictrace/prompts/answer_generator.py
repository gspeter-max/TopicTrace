"""
Final answer generation prompt: instructs the LLM to generate a precise
answer using ONLY the provided reranked context. Used for the complex query
path and the graph escalation recovery path.
"""


def build_final_answer_prompt(context_block: str) -> str:
    """
    Builds the system prompt for final answer generation,
    injecting the reranked context block into it.
    """
    return f"""You are a helpful and precise assistant. Answer the user's query using ONLY the provided context.
If the context does not contain the answer, say so clearly. Do not hallucinate external facts.

CONTEXT:
{context_block}"""
