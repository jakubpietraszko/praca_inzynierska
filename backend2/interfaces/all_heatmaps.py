from pydantic import BaseModel
from typing import List

class AllHeatmapsResponse(BaseModel):
    heatmaps: List[str]