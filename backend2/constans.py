from dataclasses import dataclass, field


@dataclass
class Constants:
    DATASETS: str = 'datasets' 
    MODELS: str = 'models'
    MODELS_DATASETS: str = 'models-datasets'
    ALL_DIRS: list[str] = field(default_factory=lambda: [
        'mask_rise',
        'mask_slic',
        'mask_seed'
    ])
    CMAP: str = 'hot'

    