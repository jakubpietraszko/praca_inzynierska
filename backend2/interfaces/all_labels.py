from pydantic import BaseModel


class AllLabelsRequest(BaseModel):
    dataset: str

class AllLabelsResponse(BaseModel):
    labels: list[str]
