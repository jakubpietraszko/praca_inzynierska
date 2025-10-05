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

class GradientStatisticsLabelManyRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int
    label: str

class GradientStatisticsLabelManyResponse(GradientResponse):
    success: bool
    list_of_rate_of_increase_in_scores: List[float]
    list_of_average_drop: List[float]

class GradientStatisticsDatasetManyRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int

class GradientStatisticsDatasetManyResponse(GradientResponse):
    success: bool
    list_of_rate_of_increase_in_scores: List[float]
    list_of_average_drop: List[float]