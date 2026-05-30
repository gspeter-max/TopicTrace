from fastapi import FastAPI
from topictrace.server.routes.research import research_router  

app = FastAPI(title = "TopicTrace") 
app.include_router(research_router)
