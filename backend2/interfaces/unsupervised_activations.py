from pydantic import BaseModel


class UnsupervisedActivationsRequest(BaseModel):
    model: str
    dataset: str
    label: str
    filename: str
    type: str

class LabelFilenameCoords(BaseModel):
    label: str
    filename: str
    coordX: float  # noqa: N815
    coordY: float  # noqa: N815

class UnsupervisedActivationsResponse(BaseModel):
    dataset: str
    label_filenames_coords: list[LabelFilenameCoords]
    success: bool
