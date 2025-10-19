
from pydantic import BaseModel

from .label_filename import LabelFilenameTarget


class GradientRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilenameTarget]
    overwrite: bool

class GradientStatisticRequest(GradientRequest):
    heatmap: str
    threshold: int
    threshold_type: str

class GradientResponse(BaseModel):
    success: bool

class ListOfScores(BaseModel):
    scores: list[float]

class GradientStatisticResponse(GradientResponse):
    softmax_scores: list[ListOfScores]

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
    list_of_rate_of_increase_in_scores: list[float]
    list_of_average_drop: list[float]

class GradientStatisticsDatasetManyRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int

class GradientStatisticsDatasetManyResponse(GradientResponse):
    success: bool
    list_of_rate_of_increase_in_scores: list[float]
    list_of_average_drop: list[float]
