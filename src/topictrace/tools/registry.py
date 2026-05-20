from topictrace.tools.web_search import web_search
from topictrace.tools.web_fetch import web_fetch
from topictrace.tools.summarize import summarize


def get_tool_definitions() -> list:
    """
    Get the tool definitions for GLM-5.1 tool calling.

    Returns a list of tool definitions in OpenAI-compatible format.
    These tell the LLM what tools are available and what parameters
    each tool expects.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for exam-related content like past papers, "
                    "syllabi, study materials, and exam tips. "
                    "Returns a list of results with title, URL, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'A-Level Biology past papers')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "Fetch a web page and convert it to clean Markdown. "
                    "Use this after web_search to get the full content of a result. "
                    "Returns the page content as Markdown text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (e.g., 'https://example.com/page')"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": (
                    "Summarize long content in relation to the user's query. "
                    "Use this after web_fetch to condense page content "
                    "into exam-relevant highlights."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The full text content to summarize"
                        },
                        "query": {
                            "type": "string",
                            "description": "The user's original question for context"
                        }
                    },
                    "required": ["content", "query"]
                }
            }
        }
    ]


def run_tool(tool_name: str, session_path: str, **kwargs):
    """
    Run a tool by name with the given arguments.

    This is the central dispatcher that maps tool names to their functions.
    The agent calls this when the LLM decides to use a tool.
    """
    tools = {
        "web_search": web_search,
        "web_fetch": web_fetch,
        "summarize": summarize,
    }

    tool_function = tools.get(tool_name)

    if tool_function is None:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(tools.keys())}"

    return tool_function(session_path=session_path, **kwargs)
