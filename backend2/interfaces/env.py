from pydantic import BaseModel


class EnvResponse(BaseModel):
    env: str
