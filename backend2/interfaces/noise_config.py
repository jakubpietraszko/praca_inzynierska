from pydantic import BaseModel


class NoiseConfig(BaseModel):
    noise_level: float
