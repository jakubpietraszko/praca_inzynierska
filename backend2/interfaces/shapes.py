from pydantic import BaseModel
from typing import List


class ShapeMetricsLabelRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int
    label: str


class ShapeMetricsLabelResponse(BaseModel):
    success: bool
    areas: List[float]
    perimeters: List[float]
    circularities: List[float]
    rectangularities: List[float]


class ShapeMetricsDatasetRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int


class ShapeMetricsDatasetResponse(BaseModel):
    success: bool
    areas: List[float]
    perimeters: List[float]
    circularities: List[float]
    rectangularities: List[float]
