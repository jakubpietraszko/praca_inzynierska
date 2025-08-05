from pydantic import BaseModel
from typing import List

class AllModelsResponse(BaseModel):
    models: List[str]