from langgraph.graph import StateGraph, START , END 
from langgraph.prebuilt import ToolNode 
from topictrace.agents.state import ResearchState
from topictrace.tools import web_fetch, web_search
from topictrace.provider.llm import get_llm_with_tools
from topictrace.prompts.research_agent import get_system_prompt
from langchain_core.messages import HumanMessage, SystemMessage


tools = [web_fetch, web_search]
llm_with_tools = get_llm_with_tools(tools)

def agent_node(state : ResearchState):
    messages = [SystemMessage(content = get_system_prompt())] + HumanMessage(state['messages']) 
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]} # because messages add to ResearchState.messages.append{.....}

def should_continue(state: ResearchState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "TOOLS"

    return END 
    
graph = StateGraph(ResearchState)
graph.add_node("AGENT", agent_node)
graph.add_node("TOOLS", ToolNode(tools))

graph.add_edge(START, "AGENT")
graph.add_conditional_edges("AGENT", should_continue)
graph.add_edge("TOOLS", "AGENT")

app = graph.compile() 
