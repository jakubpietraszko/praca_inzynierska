from pydantic import BaseModel


class HeatmapsStatisticsDatasetRequest(BaseModel):
    model: str
    dataset: str


class HeatmapsStatisticsDatasetResponse(BaseModel):
    average_drop: list[float]
    rate_of_increase: list[float]
    success: bool
