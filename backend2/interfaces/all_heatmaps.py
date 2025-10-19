from pydantic import BaseModel


class AllHeatmapsResponse(BaseModel):
    heatmaps: list[str]
