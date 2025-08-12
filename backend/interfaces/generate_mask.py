from pydantic import BaseModel
from typing import List

class LabelFilename(BaseModel):
    label: str
    filename: str

class GenerateManyMaskRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    count: int
    overwrite: bool

class GenerateMasManykResponse(BaseModel):
    success: bool

class GenerateOneMaskRequest(BaseModel):
    model: str
    dataset: str
    images: List[LabelFilename]
    overwrite: bool

class GenerateOneMaskResponse(BaseModel):
    success: bool