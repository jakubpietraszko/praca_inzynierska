from pydantic import BaseModel

from .label_filename import LabelFilenameTarget


class ListOfScores(BaseModel):
    scores: list[float]


class HeatmapRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilenameTarget]
    overwrite: bool


class HeatmapResponse(BaseModel):
    success: bool


class HeatmapStatisticRequest(HeatmapRequest):
    heatmap: str
    threshold: int
    threshold_type: str


class HeatmapStatisticResponse(HeatmapResponse):
    softmax_scores: list[ListOfScores]


class HeatmapStatisticsDatasetRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    threshold: int
    percentage: int


class HeatmapStatisticsDatasetResponse(HeatmapResponse):
    rate_of_increase_in_scores: float
    average_drop: float
    success: bool


class HeatmapStatisticsLabelRequest(HeatmapStatisticsDatasetRequest):
    label: str


class HeatmapStatisticsLabelResponse(HeatmapStatisticsDatasetResponse):
    pass


class HeatmapStatisticsLabelManyRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int
    label: str


class HeatmapStatisticsLabelManyResponse(HeatmapResponse):
    success: bool
    list_of_rate_of_increase_in_scores: list[float]
    list_of_average_drop: list[float]


class HeatmapStatisticsDatasetManyRequest(BaseModel):
    model: str
    dataset: str
    heatmap: str
    percentage: int


class HeatmapStatisticsDatasetManyResponse(HeatmapResponse):
    success: bool
    list_of_rate_of_increase_in_scores: list[float]
    list_of_average_drop: list[float]
