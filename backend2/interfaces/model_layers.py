from pydantic import BaseModel


class ModelLayersRequest(BaseModel):
    model: str

class ModelLayersResponse(BaseModel):
    layers: list[str]
