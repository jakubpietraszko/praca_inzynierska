from pydantic import BaseModel


class ActivationsRequest(BaseModel):
    model: str
    dataset: str
    label: str
    filename: str


class AnyActivationsRequest(BaseModel):
    model: str
    dataset: str
    label: str
    filename: str
    ref: str


class ActivationsResponse(BaseModel):
    success: bool
    count: int


AnyActivationsResponse = ActivationsResponse
