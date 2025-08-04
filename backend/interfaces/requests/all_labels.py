from pydantic import BaseModel

class AllLabelsRequest(BaseModel):
    dataset: str