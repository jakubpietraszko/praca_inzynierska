from pydantic import BaseModel
from typing import List

class AllsImagesRequest(BaseModel):
    dataset: str
    label: str

class AllImagesResponse(BaseModel):
    images: List[str]