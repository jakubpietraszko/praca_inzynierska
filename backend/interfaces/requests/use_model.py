from pydantic import BaseModel

class UseModelRequest(BaseModel):
    model_name: str