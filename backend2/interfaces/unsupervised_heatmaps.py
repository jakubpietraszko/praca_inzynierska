from pydantic import BaseModel

class UnsupervisedHeatmapsRequest(BaseModel):
    model: str
    dataset: str
    label: str
    heatmap: str
    type: str 


class FilenameCoords(BaseModel):
    filename: str
    coordX: float
    coordY: float

class UnsupervisedHeatmapsResponse(BaseModel):
    filenames_coords: list[FilenameCoords]
    success: bool
    
