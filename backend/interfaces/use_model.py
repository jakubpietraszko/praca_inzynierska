from pydantic import BaseModel

class UseModelRequest(BaseModel):
    model: str

class UseModelResponse(BaseModel):
    success: bool