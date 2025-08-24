from pydantic import BaseModel
from typing import List
from .label_filename import LabelFilename

class EdgeRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    overwrite: bool

class EdgeResponse(BaseModel):
    success: bool