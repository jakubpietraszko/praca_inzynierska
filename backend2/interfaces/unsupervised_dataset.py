from pydantic import BaseModel


class Coords(BaseModel):
    x: float
    y: float


class Color(BaseModel):
    r: int
    g: int
    b: int


class UnsupervisedDatasetRequest(BaseModel):
    dataset: str


class UnsupervisedDatasetItem(BaseModel):
    filename: str
    label: str
    coords: Coords
    color: Color


class UnsupervisedDatasetResponse(BaseModel):
    items: list[UnsupervisedDatasetItem]
    success: bool
