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