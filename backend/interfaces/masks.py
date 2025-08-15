from pydantic import BaseModel
from typing import List

class LabelFilename(BaseModel):
    label: str
    filename: str

class MasksRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    count: int
    overwrite: bool

class MasksResponse(BaseModel):
    success: bool