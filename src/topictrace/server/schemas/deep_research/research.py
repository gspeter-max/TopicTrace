from typing import Literal

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str
    depth: Literal["quick", "standard", "deep"] = "quick"


class ResearchResponse(BaseModel):
    answer: str
