from pydantic import BaseModel

class AllQuantitativeStatisticsRequest(BaseModel):
    model: str
    dataset: str

class AllQuantitativeStatisticsResponse(BaseModel):
    success: bool
    confusion_dict: dict[str, dict[str, int]]