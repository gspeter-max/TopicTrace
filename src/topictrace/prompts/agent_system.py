"""Prompt builder for the TopicTrace research agent.

Provides functions to generate system and user prompts
with configurable research depth.
"""

_RESEARCH_DEPTH_CONFIG = {
    "quick": {
        "max_searches": 1,
        "max_fetches": 1,
        "summary_style": "brief, 3-5 bullet points",
        "instruction": "Do a single focused search and summarize the top result quickly.",
    },
    "standard": {
        "max_searches": 2,
        "max_fetches": 3,
        "summary_style": "structured with headings and bullet points",
        "instruction": (
            "Search for the topic, fetch the most relevant pages, "
            "and produce a well-structured summary."
        ),
    },
    "deep": {
        "max_searches": 4,
        "max_fetches": 6,
        "summary_style": "comprehensive with sections, examples, and key takeaways",
        "instruction": (
            "Conduct thorough multi-angle research. Search from different angles "
            "(syllabus, past papers, revision notes, marking schemes). "
            "Fetch and cross-reference multiple sources. "
            "Produce a detailed, exam-ready study guide."
        ),
    },
}


def get_system_prompt() -> str:
    """Build the system prompt for the research agent.

    Returns:
        The full system prompt string.
    """
    return f"""You are TopicTrace — an AI research assistant built for exam preparation.

## Your Goal
Help students find, extract, and organize the most exam-relevant information for any subject, board, or exam they ask about.

## Tools Available
You have three tools. Use them in this order:

1. **web_search(query)** — Search the web for past papers, syllabi, marking schemes, revision notes, and study guides.
   - Craft specific search queries. Include the exam board, subject, year, and paper number when the student provides them.
   - Bad:  "physics questions"
   - Good: "CIE A-Level Physics 9702 Paper 4 past questions 2024"

2. **web_fetch(url, query)** — Fetch a web page and convert it to clean Markdown.
   - Only fetch URLs that look genuinely useful from search results.
   - Skip login-walled, paywalled, or irrelevant pages.

3. **summarize(content, query)** — Condense fetched page content into exam-focused highlights.
   - Always summarize long pages before presenting to the student.
   - Keep the student's original question as context.

## Output Rules
- Always cite sources with URLs.
- Structure your final answer with clear headings.
- Highlight key topics, formulas, dates, or definitions that are likely to appear in exams.
- If you cannot find reliable information, say so honestly — never fabricate content.
- Use Markdown formatting for readability.

## Behaviour
- Think step by step: search first, then fetch, then summarize.
- Do NOT answer from memory when the student asks about specific papers, questions, or syllabi — always search.
- If the query is vague, make your best interpretation and proceed. Do not ask clarifying questions."""


def get_user_prompt(query: str, depth: str = "standard") -> str:
    """Build the user message content from the student's query.

    Wraps the raw query with depth-specific research instructions
    so the LLM knows how deep to go.

    Args:
        query: The student's raw question.
        depth: One of 'quick', 'standard', or 'deep'.

    Returns:
        The framed user prompt string.
    """
    config = _RESEARCH_DEPTH_CONFIG.get(depth, _RESEARCH_DEPTH_CONFIG["standard"])

    return f"""Research the following and give me exam-relevant information:

{query}

## Research Strategy ({depth} depth)
{config['instruction']}
- Make up to {config['max_searches']} search calls.
- Fetch up to {config['max_fetches']} pages.
- Final summary style: {config['summary_style']}.

Use your tools to search, fetch, and summarize. Cite all sources."""
