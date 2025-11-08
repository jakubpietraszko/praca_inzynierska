from pydantic import BaseModel, RootModel


class ConfusionMatrixRow(RootModel[dict[str, int]]):
    pass


class ConfusionMatrix(RootModel[dict[str, ConfusionMatrixRow]]):
    pass


class ModelDatasetPredictionsRequest(BaseModel):
    model: str
    dataset: str


class ModelDatasetPredictionsResponse(BaseModel):
    predictions_proba: list[list[float]]
    predictions: list[int]
    true_labels: list[int]
    labels: list[str]
    confusion_matrix: ConfusionMatrix
    filenames: list[str]


class ModelDatasetStatisticsRequest(BaseModel):
    model: str
    dataset: str


class ModelDatasetStatisticsResponse(BaseModel):
    accuracy: float
    balanced_accuracy: float

    confusion_matrix: ConfusionMatrix
    roc_auc_per_class: dict[str, float]
    roc_auc_macro: float
    roc_auc_micro: float
    roc_auc_weighted: float

    support_per_class: dict[str, int]

    precision_per_class: dict[str, float]
    precision_macro: float
    precision_micro: float
    precision_weighted: float

    recall_per_class: dict[str, float]
    recall_macro: float
    recall_micro: float
    recall_weighted: float

    f1_score_per_class: dict[str, float]
    f1_score_macro: float
    f1_score_micro: float
    f1_score_weighted: float

    jaccard_per_class: dict[str, float]
    jaccard_macro: float
    jaccard_micro: float
    jaccard_weighted: float
