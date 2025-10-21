from pydantic import BaseModel


class ModelDatasetI(BaseModel):
    model: str
    dataset: str


class ModelDatasetLabelHeatmapI(ModelDatasetI):
    label: str
    heatmap: str


class NumeratorDenominator(BaseModel):
    numerator: dict[str, int]
    denominator: int


class Response(BaseModel):
    success: bool
