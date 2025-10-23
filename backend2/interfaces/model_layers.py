from pydantic import BaseModel


class ModelLayersRequest(BaseModel):
    model: str

class LayerConvRef(BaseModel):
    name: str
    conv: bool
    ref: str

class AllLayersRequest(BaseModel):
    model: str

class AllLayersResponse(BaseModel):
    layers: list[LayerConvRef]

class ModelLayersResponse(BaseModel):
    layers: list[str]
