from pydantic import BaseModel

class AllImagesRequest(BaseModel):
    dataset: str
    label: str