from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage 
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    messages: Annotated[
        list[AIMessage | HumanMessage], add_messages
    ]  # add_message | ], add_messages
