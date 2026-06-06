from pydantic import BaseModel
from typing import Literal

class ResearchRequest( BaseModel ):
    query : str 
    depth : Literal["quick", "standard", "deep"] = "quick"

class ResearchResponse( BaseModel ):
    answer : str 
