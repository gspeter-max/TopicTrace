import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig

from topictrace import log
from topictrace.agents.graph import app
from topictrace.agents.state import ResearchState
from topictrace.prompts import get_user_prompt
from topictrace.server.schemas.deep_research.research import (
    ResearchRequest,
    ResearchResponse,
)

research_router = APIRouter()


@research_router.post("/research", response_model=ResearchResponse)
async def research(request_input: ResearchRequest):
    try:
        state: ResearchState = {
            "messages": [
                HumanMessage(
                    content=get_user_prompt(
                        "research",
                        {"query": request_input.query, "depth": request_input.depth},
                    )
                )
            ],
        }
        config: RunnableConfig = {"configurable": {"thread_id": 1}}
        response = await app.ainvoke(state, config=config)
        return ResearchResponse(answer=response["messages"][-1].content)

    except Exception as e:
        log.error(
            "getting a error",
            error=str(e),
            exc_type=type(e).__init__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"error in research agent {e}"
        ) from e


async def streaming_yeild_generater(query: str, depth: str = "standard"):
    """ "Yields SSE-formatted tokens as LLM produces them."""

    try:
        state: ResearchState = {
            "messages": [
                HumanMessage(
                    content=get_user_prompt(
                        "research", {"query": query, "depth": depth}
                    )
                )
            ],
        }
        config: RunnableConfig = {"configurable": {"thread_id": 1}}
        async for event in app.astream_events(state, version="v2", config=config):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            if event.get("name", None) in ("web_search", "web_fetch", "summarize"):
                yield f"data: {json.dumps({'status': 'toolCalling', 'tool': event['name']})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@research_router.post("/research/stream")
async def research_stream(request_input: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        streaming_yeild_generater(request_input.query, request_input.depth),
        media_type="text/event-stream",
    )
