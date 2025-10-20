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


class ConfigService:
    def __init__(self, base_dir: str):
        from services.container import Constants, get_constants

        self.constants: Constants = get_constants()

        self.base_dir = base_dir
        self.NAME = self.constants.CONFIG_FILENAME
        self._config = None

    @property
    async def config(self) -> ConfigModel:
        if self._config is None:
            self._config = await self.load_config()
        return self._config

    async def load_config(self) -> ConfigModel:
        import json
        import os

        path = os.path.join(self.base_dir, self.NAME)
        if not os.path.exists(path):
            return await self._get_default_config()

        with open(path) as f:
            data = json.load(f)
            methods = {}
            for method_name, method_params in data.get("methods", {}).items():
                methods[method_name] = XAIMethodConfig(**method_params)
            return ConfigModel(methods=methods)

    async def _get_default_config(self) -> ConfigModel:
        default_methods = {
            self.constants.GRAD_CAM: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.GRAD_CAM_PLUS_PLUS: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.GROUP_CAM: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                group_size=self.constants.DEFAULT_GROUP_SIZE,
                k_size=self.constants.DEFAULT_KSIZE,
                sigma=self.constants.DEFAULT_SIGMA,
            ),
            self.constants.HIRES_CAM: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.WGRAD_CAM: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.ABLATION_CAM: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.XGRAD_CAM: XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON),
            self.constants.SMOOTH_GRADIENT: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                n_samples=self.constants.DEFAULT_N_SAMPLES,
                noise_level=self.constants.DEFAULT_NOISE_LEVEL,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.SMOOTH_GRADIENT_SQUARE: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                n_samples=self.constants.DEFAULT_N_SAMPLES,
                noise_level=self.constants.DEFAULT_NOISE_LEVEL,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.VAR_GRADIENT: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                n_samples=self.constants.DEFAULT_N_SAMPLES,
                noise_level=self.constants.DEFAULT_NOISE_LEVEL,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.GRADIENT: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.GRADIENT_SQUARED: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.DECONVOLUTION: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.GUIDED_BACKPROPAGATION: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.INPUT_GRADIENT: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.INTEGRATED_GRADIENT: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                baseline=self.constants.DEFAULT_BASELINE,
                steps=self.constants.DEFAULT_STEPS,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.XRAI: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                aggregation=self.constants.DEFAULT_AGGREGATION,
                baseline=self.constants.DEFAULT_BASELINE,
                super_pixels=self.constants.DEFAULT_SUPER_PIXELS,
                segmentation=self.constants.DEFAULT_SEGMENTATION,
            ),
            self.constants.KERNEL_SHAP: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                n_samples=self.constants.DEFAULT_N_SAMPLES,
                aggregation=self.constants.DEFAULT_AGGREGATION,
                baseline=self.constants.DEFAULT_BASELINE,
                super_pixels=self.constants.DEFAULT_SUPER_PIXELS,
                segmentation=self.constants.DEFAULT_SEGMENTATION,
            ),
            self.constants.LIME: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                n_samples=self.constants.DEFAULT_N_SAMPLES,
                aggregation=self.constants.DEFAULT_AGGREGATION,
                baseline=self.constants.DEFAULT_BASELINE,
                super_pixels=self.constants.DEFAULT_SUPER_PIXELS,
                segmentation=self.constants.DEFAULT_SEGMENTATION,
            ),
            self.constants.LRP_EPSILON: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON, aggregation=self.constants.DEFAULT_AGGREGATION
            ),
            self.constants.LRP_GAMMA: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                gamma=self.constants.DEFAULT_GAMMA,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.LRP_ALPHA_BETA: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON,
                alpha=self.constants.DEFAULT_ALPHA,
                beta=self.constants.DEFAULT_BETA,
                aggregation=self.constants.DEFAULT_AGGREGATION,
            ),
            self.constants.LRP_ZPLUS: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON, aggregation=self.constants.DEFAULT_AGGREGATION
            ),
            self.constants.LRP_W2: XAIMethodConfig(
                epsilon=self.constants.DEFAULT_EPSILON, aggregation=self.constants.DEFAULT_AGGREGATION
            ),
        }
        return ConfigModel(methods=default_methods)

    async def save_config(self, config: ConfigModel):
        import json
        import os

        path = os.path.join(self.base_dir, self.NAME)
        config_dict = {
            "methods": {
                method_name: method_config.model_dump(exclude_none=True)
                for method_name, method_config in config.methods.items()
            }
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)

        self._config = config

    async def get_method_config(self, method_name: str) -> XAIMethodConfig:
        config = await self.config
        if method_name not in config.methods:
            default_config = await self._get_default_config()
            if method_name in default_config.methods:
                return default_config.methods[method_name]
            else:
                return XAIMethodConfig(epsilon=self.constants.DEFAULT_EPSILON)
        return config.methods[method_name]

    async def update_method_config(self, method_name: str, method_config: XAIMethodConfig):
        current_config = await self.config
        current_config.methods[method_name] = method_config
        await self.save_config(current_config)

    async def get_all_method_names(self) -> list[str]:
        config = await self.config
        return list(config.methods.keys())
