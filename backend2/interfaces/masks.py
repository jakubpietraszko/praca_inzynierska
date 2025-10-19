from pydantic import BaseModel

from .label_filename import LabelFilename


class MasksRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilename]
    count: int
    overwrite: bool

class MasksResponse(BaseModel):
    success: bool
