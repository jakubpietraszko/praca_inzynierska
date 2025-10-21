from pydantic import BaseModel


class UnsupervisedHeatmapsRequest(BaseModel):
    model: str
    dataset: str
    label: str
    heatmap: str
    type: str


class FilenameCoords(BaseModel):
    filename: str
    coordX: float  # noqa: N815
    coordY: float  # noqa: N815

class UnsupervisedHeatmapsResponse(BaseModel):
    filenames_coords: list[FilenameCoords]
    success: bool

