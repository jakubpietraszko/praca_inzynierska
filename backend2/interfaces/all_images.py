from pydantic import BaseModel


class AllsImagesRequest(BaseModel):
    dataset: str
    label: str

class AllImagesResponse(BaseModel):
    images: list[str]
