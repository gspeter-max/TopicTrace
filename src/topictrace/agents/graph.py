from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph  # type: ignore[import] 
from langgraph.graph.state import CompiledStateGraph  #type: ignore[import] 
from langgraph.prebuilt import ToolNode

from topictrace.agents.state import ResearchState
from topictrace.db.postgres.client import pool
from topictrace.prompts import get_system_prompt, get_user_prompt
from topictrace.provider.llm import get_llm, get_llm_with_tools
from topictrace.tools import web_fetch, web_search

tools = [web_fetch, web_search]
llm_with_tools = get_llm_with_tools(tools)
llm = get_llm("MISTRAL_AI")


async def get_memory(session_id: str) -> str | None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT memory_information FROM memory WHERE session_id = %s ORDER BY session_id DESC LIMIT 1""",
                (session_id,),
            )
            row = cur.fetchone()
            return row["memory_information"] if row else None


async def store_in_memory(session_id: str, response: AIMessage) -> str:
    compact_message = [
        SystemMessage(content=get_system_prompt("compact")),
        HumanMessage(
            content=get_user_prompt("compact", {"response_text": response.content})
        ),
    ]
    compact_response = await llm.ainvoke(compact_message)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memory (session_id, memory_information) VALUES (%s, %s)""",
                (session_id, compact_response.content),
            )
            return "[DONE]"


async def agent_node(state: ResearchState) -> dict[str, Any]:
    memory = await get_memory(session_id="session_id_123")
    messages = [
        SystemMessage(content=get_system_prompt("research")),
        HumanMessage(
            content=f" [MEMORY CONTENT] {memory} [USER MESSAGE] {state['messages']}"
        ),
    ]

    config: RunnableConfig = {"configurable": {"thread_id": 1}}
    response = llm_with_tools.invoke(messages, config)
    return {
        "messages": [response]
    }  # because messages add to ResearchState.messages.append{.....}


async def should_continue(state: ResearchState) -> str:
    try:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "TOOLS"
        else:
            _ = await store_in_memory(
                session_id="session_id_123", response=cast(AIMessage, last_message)
            )
            return END

    except Exception as e:
        raise RuntimeError(f"[GRAPH] [SHOULD_CONTINUE] : {e}")


async def deepResearchGraph() -> CompiledStateGraph[
    ResearchState, None, ResearchState, ResearchState
]:
    graph = (
        StateGraph(ResearchState)
        .add_node("AGENT", agent_node)
        .add_node("TOOLS", ToolNode(tools))
        .add_edge(START, "AGENT")
        .add_conditional_edges("AGENT", should_continue)
        .add_edge("TOOLS", "AGENT")
    )

    return graph.compile()  # pyright: ignore[reportUnknownMemberType]
