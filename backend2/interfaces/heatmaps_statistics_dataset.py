from pydantic import BaseModel
from typing import List


class HeatmapsStatisticsDatasetRequest(BaseModel):
    model: str
    dataset: str


class HeatmapsStatisticsDatasetResponse(BaseModel):
    average_drop: List[float]
    rate_of_increase: List[float]
    success: bool
