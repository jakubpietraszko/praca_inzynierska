from pydantic import BaseModel

class ListOfStringsResponse(BaseModel):
    strings: list[str]