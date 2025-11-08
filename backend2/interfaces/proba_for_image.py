from pydantic import BaseModel

class ProbaForImageRequest(BaseModel):
    model: str
    dataset: str
    label: str
    filename: str

class ProbaForImageResponse(BaseModel):
    probabilities: list[float]
    classes: list[str]