from pydantic import BaseModel

class ClassesRequest(BaseModel):
    classes: list[str]