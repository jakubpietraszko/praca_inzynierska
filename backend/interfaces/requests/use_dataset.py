from pydantic import BaseModel

class UseDatasetRequest(BaseModel):
    dataset_name: str