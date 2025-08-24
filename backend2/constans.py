from dataclasses import dataclass, field


@dataclass
class Constants:
    DATASETS: str = 'datasets' 
    MODELS: str = 'models'
    MODELS_DATASETS: str = 'models-datasets'

    MASK_RISE: str = 'mask_rise'
    MASK_SLIC: str = 'mask_slic'
    MASK_SEED: str = 'mask_seed'

    ACTIVATIONS: str = 'activations'

    GRADIENT: str = 'gradient'
    DECONVOLUTION: str = 'deconvolution'
    GUIDED_BACKPROPAGATION: str = 'guided_backpropagation'
    INTEGRATED_GRADIENT: str = 'integrated_gradient'

    SMOOTH_GRADIENT: str = 'smooth_gradient'
    SMOOTH_GRADIENT_SQUARE: str = 'smooth_gradient_square'
    VAR_GRADIENT: str = 'var_gradient'

    GRAD_CAM: str = 'grad_cam'
    GRAD_CAM_PLUS_PLUS: str = 'grad_cam_plus_plus'
    HIRES_CAM: str = 'hires_cam'

    ALL_DIRS: list[str] = field(default_factory=lambda: [
        Constants.MASK_RISE,
        Constants.MASK_SLIC,
        Constants.MASK_SEED,

        Constants.ACTIVATIONS,

        Constants.GRADIENT,
        Constants.DECONVOLUTION,
        Constants.GUIDED_BACKPROPAGATION,
        Constants.INTEGRATED_GRADIENT,

        Constants.SMOOTH_GRADIENT,
        Constants.SMOOTH_GRADIENT_SQUARE,
        Constants.VAR_GRADIENT,
        
        Constants.GRAD_CAM,
        Constants.GRAD_CAM_PLUS_PLUS,
        Constants.HIRES_CAM,
    ])

    CMAP: str = 'hot',
    EPSILON: float = 1e-8
    BASELINE: int = 0
    STEPS: int = 50
    NOISE_LEVEL: float = 0.1
    NUM_SAMPLES: int = 50

    