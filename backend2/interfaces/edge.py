from pydantic import BaseModel

from .label_filename import LabelFilename


class EdgeRequest(BaseModel):
    model: str
    dataset: str
    images: list[LabelFilename]
    overwrite: bool

class EdgeResponse(BaseModel):
    success: bool
