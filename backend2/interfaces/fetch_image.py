from pydantic import BaseModel

class FetchImageRequest(BaseModel):
    dataset: str
    label: str
    filename: str

class FetchImageResponse(BaseModel):
    image: str