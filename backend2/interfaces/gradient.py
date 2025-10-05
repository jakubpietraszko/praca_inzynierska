from pydantic import BaseModel
from typing import List
from .label_filename import LabelFilenameTarget

class GradientRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilenameTarget]
    overwrite: bool

class GradientStatisticRequest(GradientRequest):
    heatmap: str
    threshold: int
    threshold_type: str

class GradientResponse(BaseModel):
    success: bool

class ListOfScores(BaseModel):
    scores: List[float]

class GradientStatisticResponse(GradientResponse):
    softmax_scores: List[ListOfScores]

class GradientStatisticsDatasetRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    threshold: int
    percentage: int

class GradientStatisticsLabelRequest(GradientStatisticsDatasetRequest):
    label: str

class GradientStatisticsDatasetResponse(GradientResponse):
    rate_of_increase_in_scores: float
    average_drop: float
    success: bool