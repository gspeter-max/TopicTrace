from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langgraph.graph.message import AnyMessage


class ResearchState(TypedDict):
    messages: Annotated[
        list[AnyMessage], add_messages
    ]  # add_message message not replace
