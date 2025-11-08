from pydantic import BaseModel

from .label_filename import LabelFilenameTarget


class ListOfScores(BaseModel):
    scores: list[float]


class InsertionDeletionRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilenameTarget]
    heatmap: str
    threshold: int


class InsertionDeletionResponse(BaseModel):
    insertion_scores: list[ListOfScores]
    deletion_scores: list[ListOfScores]
    insertion_auc: float
    deletion_auc: float
