from pydantic import BaseModel
from typing import List
from .label_filename import LabelFilename

class MasksRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    count: int
    overwrite: bool

class MasksResponse(BaseModel):
    success: bool