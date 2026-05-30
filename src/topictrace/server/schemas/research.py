from pydantic import BaseModel

class ResearchRequest( BaseModel ):
    query : str 
    depth : str = "standard"

class ResearchResponse( BaseModel ):
    answer : str 
