from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from topictrace.server.routes.research import research_router  
from fastapi.middleware.cors import CORSMiddleware
import time 

app = FastAPI(title = "TopicTrace") 
app.include_router(research_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

default_dict = defaultdict(list)

@app.middleware(middleware_type= "http")
async def rate_limit(request: Request, call_next):
    client_host = request.client.host 
    now = time.time()  
    
    default_dict[client_host] = [ rt for rt in default_dict[client_host] if (now - rt) < 60]
    if len(default_dict[client_host]) >= 10:
        raise HTTPException(status_code=429, detail="rate limit exceeds")
    
    default_dict[client_host].append(now)
    return await call_next(request) 

@app.get("/health/live")
async def health_check() -> dict:
    return {"status" : "ok"}

