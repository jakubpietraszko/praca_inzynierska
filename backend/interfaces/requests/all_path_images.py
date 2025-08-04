from pydantic import BaseModel

class AllPathsImages(BaseModel):
    dataset: str
    label: str