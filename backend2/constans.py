from dataclasses import dataclass, field


@dataclass
class Constants:
    DATASETS: str = 'datasets'
    MODELS: str = 'models'
    MODELS_DATASETS: str = 'models-datasets'

    MASK_RISE: str = 'mask_rise'
    MASK_SLIC: str = 'mask_slic'
    MASK_SEED: str = 'mask_seed'

    EDGE_CANNY: str = 'edge_canny'

    ACTIVATIONS: str = 'activations'

    GRADIENT: str = 'gradient'
    DECONVOLUTION: str = 'deconvolution'
    GUIDED_BACKPROPAGATION: str = 'guided_backpropagation'
    INPUT_GRADIENT: str = 'input_gradient'

    INTEGRATED_GRADIENT: str = 'integrated_gradient'
    SMOOTH_INTEGRATED_GRADIENT: str = 'smooth_integrated_gradient'
    EXPECTED_GRADIENT: str = 'expected_gradient'

    SMOOTH_GRADIENT: str = 'smooth_gradient'
    SMOOTH_GRADIENT_SQUARE: str = 'smooth_gradient_square'
    VAR_GRADIENT: str = 'var_gradient'

    GRAD_CAM: str = 'grad_cam'
    GRAD_CAM_PLUS_PLUS: str = 'grad_cam_plus_plus'
    WGRAD_CAM: str = 'wgrad_cam'
    HIRES_CAM: str = 'hires_cam'
    ABLATION_CAM: str = 'ablation_cam'
    GROUP_CAM: str = 'group_cam'
    SCORE_CAM: str = 'score_cam'
    XGRAD_CAM: str = 'xgrad_cam'
    EIGEN_CAM: str = 'eigen_cam'

    XRAI: str = 'xrai'
    EXACT_SHAP: str = 'exact_shap'
    KERNEL_SHAP: str = 'kernel_shap'
    RISE: str = 'rise'

    LIME: str = 'lime'

    LRP: str = 'lrp_zero'
    LRP_EPSILON: str = 'lrp_epsilon'
    LRP_GAMMA: str = 'lrp_gamma'
    LRP_ALPHA_BETA: str = 'lrp_alpha_beta'
    LRP_FLAT: str = 'lrp_flat'

    DEEP_TAYLOR: str = 'deep_taylor'

    LIFT: str = 'lift'
    DEEP_LIFT: str = 'deep_lift'

    ALL_DIRS: list[str] = field(default_factory=lambda: [
        Constants.MASK_RISE,
        Constants.MASK_SLIC,
        Constants.MASK_SEED,

        Constants.EDGE_CANNY,

        Constants.ACTIVATIONS,

        Constants.GRADIENT,
        Constants.DECONVOLUTION,
        Constants.GUIDED_BACKPROPAGATION,
        Constants.INPUT_GRADIENT,

        Constants.INTEGRATED_GRADIENT,
        Constants.SMOOTH_INTEGRATED_GRADIENT,
        Constants.EXPECTED_GRADIENT,

        Constants.SMOOTH_GRADIENT,
        Constants.SMOOTH_GRADIENT_SQUARE,
        Constants.VAR_GRADIENT,

        Constants.GRAD_CAM,
        Constants.GRAD_CAM_PLUS_PLUS,
        Constants.HIRES_CAM,

        Constants.LRP,
        Constants.LRP_EPSILON,
        Constants.LRP_GAMMA,
        Constants.LRP_ALPHA_BETA,
        Constants.LRP_FLAT,
        Constants.DEEP_TAYLOR,
    ])

    CMAP: str = 'hot',
    EPSILON: float = 1e-8
    BASELINE: int = 0
    STEPS: int = 50
    NOISE_LEVEL: float = 0.1
    NUM_SAMPLES: int = 50
    AGGREGATION: str = 'max'  # mean, sum, max
