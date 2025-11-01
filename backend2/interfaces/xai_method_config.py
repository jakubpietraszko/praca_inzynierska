from pydantic import BaseModel, Field


class XAIMethodConfig(BaseModel):
    epsilon: float | None = None
    aggregation: str | None = None
    group_size: int | None = None
    k_size: int | None = None
    sigma: float | None = None
    n_samples: int | None = None
    noise_level: float | None = None
    baseline: float | None = None
    steps: int | None = None
    super_pixels: int | None = None
    segmentation: str | None = None
    gamma: float | None = None
    alpha: float | None = None
    beta: float | None = None


class ConfigModel(BaseModel):
    methods: dict[str, XAIMethodConfig] = Field(default_factory=dict)
