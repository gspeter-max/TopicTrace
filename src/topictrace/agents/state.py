from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    messages : Annotated[list, add_messages] # add_message message not replace 
    session_path : str 


