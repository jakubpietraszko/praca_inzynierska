from pydantic import BaseModel
from typing import List
from .label_filename import LabelFilenameTarget

class GradientRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilenameTarget]
    overwrite: bool

class GradientResponse(BaseModel):
    success: bool