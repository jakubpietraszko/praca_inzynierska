from pydantic import BaseModel

class UseDatasetRequest(BaseModel):
    dataset: str

class UseDatasetResponse(BaseModel):
    success: bool