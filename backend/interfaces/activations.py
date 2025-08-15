from pydantic import BaseModel
from typing import List

class LabelFilename(BaseModel):
    label: str
    filename: str

class ActivationsRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]

class ActivationsResponse(BaseModel):
    success: bool