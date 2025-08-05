from pydantic import BaseModel
from typing import List

class AllLabelsRequest(BaseModel):
    dataset: str

class AllLabelsResponse(BaseModel):
    labels: List[str]