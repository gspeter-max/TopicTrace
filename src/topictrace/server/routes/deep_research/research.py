from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from topictrace.server.schemas.deep_research.research import ResearchRequest, ResearchResponse
from topictrace.prompts.research_agent import get_user_prompt
from langchain_core.messages import HumanMessage
from topictrace.agents.graph import app
from topictrace import log
import uuid
import json
import asyncio

research_router = APIRouter()

@research_router.post("/research", response_model= ResearchResponse)
async def research( request_input : ResearchRequest):
    try:
        state = {
            "messages" : [HumanMessage(content = get_user_prompt(request_input.query, request_input.depth))],
        }
        response = await app.ainvoke(state)
        return ResearchResponse(answer = response['messages'][-1].content)
    except Exception as e:
        log.exception(f'gettings error : {e}')
        raise HTTPException(
            status_code = 500,
            detail= f"error in research agent {e}"
        ) from e

async def streaming_yeild_generater(query: str, depth: str = "standard"):
    """"Yields SSE-formatted tokens as LLM produces them."""

    try:
        state = {
            "messages" : [HumanMessage(content = get_user_prompt(query, depth))],
        }
        async for event in app.astream_events(state , version="v2"):
            if event["event"] == "on_chat_model_stream":
                token = event['data']["chunk"].content 
                if token:
                    yield f"data: {json.dumps({'token' : token})}\n\n"
                
            if event.get("name", None) in ("web_search", "web_fetch", "summarize"):
                yield f"data: {json.dumps({'status': 'toolCalling','tool' : event['name']})}\n\n"
    
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error','error' : str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@research_router.post("/research/stream")
async def research_stream(request_input : ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        streaming_yeild_generater(request_input.query, request_input.depth),
        media_type= "text/event-stream"
    )
