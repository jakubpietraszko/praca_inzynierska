from pydantic import BaseModel

class AllPathsImagesRequest(BaseModel):
    dataset: str
    label: str