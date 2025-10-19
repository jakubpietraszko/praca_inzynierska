from pydantic import BaseModel


class InitDirRequest(BaseModel):
    model: str
    dataset: str

class InitDirResponse(BaseModel):
    success: bool
