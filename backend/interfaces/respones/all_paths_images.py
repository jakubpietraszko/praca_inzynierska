from pydantic import BaseModel
from typing import List

class AllPathsImagesResponse(BaseModel):
    pathsImages: List[str]