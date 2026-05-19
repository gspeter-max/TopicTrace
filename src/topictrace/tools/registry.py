"""
Tool registry for TopicTrace.

Provides:
1. get_tool_definitions() — returns OpenAI-compatible tool definitions for LLM
2. run_tool() — dispatches tool calls to the correct function

The LLM uses get_tool_definitions() to know what tools are available.
When the LLM decides to use a tool, run_tool() executes it.
"""

from topictrace.tools.web_search import web_search
from topictrace.tools.web_fetch import web_fetch
from topictrace.tools.summarize import summarize


def get_tool_definitions() -> list:
    """
    Get the tool definitions for GLM-5.1 tool calling.

    Returns a list of tool definitions in OpenAI-compatible format.
    These tell the LLM what tools are available and what parameters
    each tool expects.

    Returns:
        List of tool definition dicts in OpenAI function calling format
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

    Args:
        tool_name: Name of the tool to run ("web_search", "web_fetch", or "summarize")
        session_path: Path to the current session directory
        **kwargs: Arguments to pass to the tool function

    Returns:
        The result from the tool function, or an error string if tool is unknown
    """
    # Map tool names to their actual functions
    tools = {
        "web_search": web_search,
        "web_fetch": web_fetch,
        "summarize": summarize,
    }

    # Look up the tool function
    tool_function = tools.get(tool_name)

    if tool_function is None:
        # Return error message for unknown tools
        return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(tools.keys())}"

    # Call the tool with session_path and the provided arguments
    return tool_function(session_path=session_path, **kwargs)
