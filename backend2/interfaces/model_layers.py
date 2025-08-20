from pydantic import BaseModel
from typing import List

class ModelLayersRequest(BaseModel):
    model: str

class ModelLayersResponse(BaseModel):
    layers: List[str]