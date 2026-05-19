"""
Tool registry for TopicTrace.

Provides a central place to run any tool by name.
The agent can call run_tool("web_search", query="...") without
needing to import each tool directly.
"""

from topictrace.tools.web_search import web_search
from topictrace.tools.web_fetch import web_fetch
from topictrace.tools.summarize import summarize


def run_tool(tool_name: str, **kwargs):
    """
    Run a tool by name with the given arguments.

    This is the central dispatcher that maps tool names to their functions.
    The agent calls this instead of importing each tool directly.

    Args:
        tool_name: Name of the tool to run ("web_search", "web_fetch", or "summarize")
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

    # Call the tool with the provided arguments
    return tool_function(**kwargs)
