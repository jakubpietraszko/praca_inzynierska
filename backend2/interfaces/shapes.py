from pydantic import BaseModel


class ShapeMetricsLabelRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int
    label: str


class ShapeMetricsLabelResponse(BaseModel):
    success: bool
    areas: list[float]
    perimeters: list[float]
    circularities: list[float]
    rectangularities: list[float]


class ShapeMetricsDatasetRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int


class ShapeMetricsDatasetResponse(BaseModel):
    success: bool
    areas: list[float]
    perimeters: list[float]
    circularities: list[float]
    rectangularities: list[float]
