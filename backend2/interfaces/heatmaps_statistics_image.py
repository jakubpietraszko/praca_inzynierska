from pydantic import BaseModel


class HeatmapsStatisticsImageRequest(BaseModel):
    model: str
    dataset: str
    label: str
    heatmap: str
    image: str

class HeatmapsStatisticsImageResponse(BaseModel):
    deletion: float
    insertion: float
    overall: float
    success: bool