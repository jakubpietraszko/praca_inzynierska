from pydantic import BaseModel

from .label_filename import LabelFilename


class SuperPixelsRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilename]
    superPixels: int  # noqa: N815
    overwrite: bool

class SuperPixelsResponse(BaseModel):
    success: bool
    superPixels: list[int]  # noqa: N815
