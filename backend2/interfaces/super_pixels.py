from pydantic import BaseModel
from typing import List
from .label_filename import LabelFilename

class SuperPixelsRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    superPixels: int
    overwrite: bool

class SuperPixelsResponse(BaseModel):
    success: bool
    superPixels: List[int]