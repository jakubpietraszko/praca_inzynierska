from pydantic import BaseModel


class AllModelsResponse(BaseModel):
    models: list[str]
