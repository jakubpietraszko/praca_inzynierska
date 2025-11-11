from pydantic import BaseModel


class Coords(BaseModel):
    x: float
    y: float


class Color(BaseModel):
    r: int
    g: int
    b: int


class UnsupervisedLabelRequest(BaseModel):
    dataset: str
    label: str


class UnsupervisedLabelItem(BaseModel):
    filename: str
    coords: Coords
    color: Color


class UnsupervisedLabelResponse(BaseModel):
    items: list[UnsupervisedLabelItem]
    success: bool
