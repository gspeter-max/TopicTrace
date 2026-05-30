from fastapi import APIRouter, HTTPException
from topictrace.server.schemas.research import ResearchRequest, ResearchResponse
from topictrace.prompts.agent_system import get_user_prompt
from topictrace.session import create_session
from langchain_core.messages import HumanMessage
from topictrace.agents.graph import app 
from topictrace import log 


research_router = APIRouter() 

@research_router.post("/research", response_model= ResearchResponse)
async def research( request_input : ResearchRequest):
    try:
        session_path =  create_session(request_input.query[:50])
        state = {
            "messages" : [HumanMessage(content = get_user_prompt(request_input.query))],
            "session_path" : session_path
        }
        response = await app.ainvoke(state)
        return ResearchResponse(answer = response['messages'][-1].content)
    except Exception as e:
        log.exception(f'gettings error : {e}')
        raise HTTPException(
            status_code = 500, 
            detail= f"error in research agent {e}"
        ) from e 

