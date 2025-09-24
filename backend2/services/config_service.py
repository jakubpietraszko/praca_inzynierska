from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class XAIMethodConfig(BaseModel):
    epsilon: Optional[float] = None
    aggregation: Optional[str] = None
    group_size: Optional[int] = None
    k_size: Optional[int] = None
    sigma: Optional[float] = None
    n_samples: Optional[int] = None
    noise_level: Optional[float] = None
    baseline: Optional[float] = None
    steps: Optional[int] = None
    super_pixels: Optional[int] = None
    segmentation: Optional[str] = None


class ConfigModel(BaseModel):
    methods: Dict[str, XAIMethodConfig] = Field(default_factory=dict)


class ConfigService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.NAME = 'config.json'
        self._config = None

    @property
    async def config(self) -> ConfigModel:
        if self._config is None:
            self._config = await self.load_config()
        return self._config

    async def load_config(self) -> ConfigModel:
        import os
        import json

        path = os.path.join(self.base_dir, self.NAME)
        if not os.path.exists(path):
            return await self._get_default_config()

        with open(path, 'r') as f:
            data = json.load(f)
            methods = {}
            for method_name, method_params in data.get('methods', {}).items():
                methods[method_name] = XAIMethodConfig(**method_params)
            return ConfigModel(methods=methods)

    async def _get_default_config(self) -> ConfigModel:
        default_methods = {
            "grad_cam": XAIMethodConfig(epsilon=1e-6),
            "grad_cam_plus_plus": XAIMethodConfig(epsilon=1e-6),
            "group_cam": XAIMethodConfig(epsilon=1e-6, group_size=8, k_size=5, sigma=0.5),
            "hires_cam": XAIMethodConfig(epsilon=1e-6),
            "wgrad_cam": XAIMethodConfig(epsilon=1e-6),
            "ablation_cam": XAIMethodConfig(epsilon=1e-6),
            "xgrad_cam": XAIMethodConfig(epsilon=1e-6),

            "smooth_grad": XAIMethodConfig(epsilon=1e-6, n_samples=50, noise_level=0.1, aggregation="max"),
            "smooth_grad_squared": XAIMethodConfig(epsilon=1e-6, n_samples=50, noise_level=0.1, aggregation="max"),
            "var_grad": XAIMethodConfig(epsilon=1e-6, n_samples=50, noise_level=0.1, aggregation="max"),

            "gradient": XAIMethodConfig(epsilon=1e-6, aggregation="max"),
            "gradient_squared": XAIMethodConfig(epsilon=1e-6, aggregation="max"),
            "deconvolution": XAIMethodConfig(epsilon=1e-6, aggregation="max"),
            "guided_backpropagation": XAIMethodConfig(epsilon=1e-6, aggregation="max"),
            "input_gradient": XAIMethodConfig(epsilon=1e-6, aggregation="max"),

            "integrated_gradient": XAIMethodConfig(epsilon=1e-6, baseline=0, steps=50, aggregation="max"),

            "xrai": XAIMethodConfig(epsilon=1e-6, aggregation="max", baseline=0, super_pixels=30, segmentation="seed"),
            "kernel_shap": XAIMethodConfig(epsilon=1e-6, n_samples=50, aggregation="max", baseline=0, super_pixels=30, segmentation="seed"),
            "lime": XAIMethodConfig(epsilon=1e-6, n_samples=50, aggregation="max", baseline=0, super_pixels=30, segmentation="seed")

        }
        return ConfigModel(methods=default_methods)

    async def save_config(self, config: ConfigModel):
        import os
        import json

        path = os.path.join(self.base_dir, self.NAME)
        config_dict = {
            "methods": {
                method_name: method_config.model_dump(exclude_none=True)
                for method_name, method_config in config.methods.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        self._config = config

    async def get_method_config(self, method_name: str) -> XAIMethodConfig:
        config = await self.config
        if method_name not in config.methods:
            default_config = await self._get_default_config()
            if method_name in default_config.methods:
                return default_config.methods[method_name]
            else:
                return XAIMethodConfig(epsilon=1e-6)
        return config.methods[method_name]

    async def update_method_config(self, method_name: str, method_config: XAIMethodConfig):
        current_config = await self.config
        current_config.methods[method_name] = method_config
        await self.save_config(current_config)

    async def get_all_method_names(self) -> list[str]:
        config = await self.config
        return list(config.methods.keys())
