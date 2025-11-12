from pydantic import BaseModel


class HeatmapNotification(BaseModel):
    dataset: str
    model: str
    label: str
    heatmap_type: str
    filename: str
