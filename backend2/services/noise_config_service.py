from interfaces.noise_config import NoiseConfig


class NoiseConfigService:
    def __init__(self, base_dir: str) -> None:
        from services.container import Constants, get_constants

        self.constants: Constants = get_constants()
        self.NAME = self.constants.NOISE_CONFIG_FILENAME
        self._noise_config: NoiseConfig | None = None
        self.base_dir = base_dir

    @property
    async def noise_config(self) -> NoiseConfig:
        if self._noise_config is None:
            self._noise_config = await self.load_noise_config()
        return self._noise_config

    async def load_noise_config(self) -> NoiseConfig:
        import json
        import os

        path = os.path.join(self.base_dir, self.NAME)
        if not os.path.exists(path):
            return await self._get_default_noise_config()

        with open(path) as f:
            data = json.load(f)
            return NoiseConfig(**data)

    async def _get_default_noise_config(self) -> NoiseConfig:
        return NoiseConfig(noise_level=self.constants.DEFAULT_NOISE_LEVEL)

    async def save_noise_config(self, noise_config: NoiseConfig) -> None:
        import json
        import os

        path = os.path.join(self.base_dir, self.NAME)
        data = noise_config.model_dump()

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        self._noise_config = noise_config

    async def get_noise_level(self) -> float:
        noise_config = await self.noise_config
        return noise_config.noise_level

    async def update_noise_level(self, noise_level: float) -> None:
        new_config = NoiseConfig(noise_level=noise_level)
        await self.save_noise_config(new_config)
