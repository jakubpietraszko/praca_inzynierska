from pydantic import BaseModel

class ListOfStringsResponse(BaseModel):
    labels: list[str]