from dataclasses import dataclass, field


@dataclass
class Constants:
    DATASETS: str = "datasets"
    MODELS: str = "models"
    MODELS_DATASETS: str = "models-datasets"

    STATISTICS: str = "statistics"

    INSERTION: str = "insertion"
    DELETION: str = "deletion"
    OVERALL: str = "overall"
    AVERAGE_DROP: str = "average_drop"
    RATE_OF_INCREASE_IN_SCORES: str = "rate_of_increase_in_scores"
    STATISTICS_TYPES: list[str] = field(default_factory=lambda: [])
    STATISTICS_PERCENTAGES_STEP: list[int] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    MASK_RISE: str = "mask_rise"
    MASK_SLIC: str = "mask_slic"
    MASK_SEED: str = "mask_seed"

    EDGE_CANNY: str = 'edge_canny'

    ACTIVATIONS: str = "activations"

    GRADIENT: str = "gradient"
    GRADIENT_SQUARED: str = "gradient_squared"
    DECONVOLUTION: str = "deconvolution"
    GUIDED_BACKPROPAGATION: str = "guided_backpropagation"
    INPUT_GRADIENT: str = "input_gradient"

    INTEGRATED_GRADIENT: str = "integrated_gradient"

    SMOOTH_GRADIENT: str = "smooth_gradient"
    SMOOTH_GRADIENT_SQUARE: str = "smooth_gradient_square"
    VAR_GRADIENT: str = "var_gradient"

    GRAD_CAM: str = "grad_cam"
    GRAD_CAM_PLUS_PLUS: str = "grad_cam_plus_plus"
    WGRAD_CAM: str = "wgrad_cam"
    HIRES_CAM: str = "hires_cam"
    ABLATION_CAM: str = "ablation_cam"
    GROUP_CAM: str = "group_cam"
    XGRAD_CAM: str = "xgrad_cam"
    # SCORE_CAM: str = 'score_cam'
    # EIGEN_CAM: str = 'eigen_cam'

    XRAI: str = "xrai"
    KERNEL_SHAP: str = "kernel_shap"
    LIME: str = "lime"
    # EXACT_SHAP: str = 'exact_shap'
    # RISE: str = 'rise'

    LRP_EPSILON: str = "lrp_epsilon"
    LRP_GAMMA: str = "lrp_gamma"
    LRP_ALPHA_BETA: str = "lrp_alpha_beta"
    LRP_ZPLUS: str = "lrp_zplus"
    LRP_W2: str = "lrp_w2"

    AVAILABLE_HEATMAPS: list[str] = field(default_factory=lambda: [])

    ALL_DIRS: list[str] = field(default_factory=lambda: [])

    XAI_TRANSLATIONS: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.AVAILABLE_HEATMAPS = [
            self.GRADIENT,
            # self.GRADIENT_SQUARED,
            self.DECONVOLUTION,
            self.GUIDED_BACKPROPAGATION,
            self.INPUT_GRADIENT,
            self.INTEGRATED_GRADIENT,
            self.SMOOTH_GRADIENT,
            # self.SMOOTH_GRADIENT_SQUARE,
            self.VAR_GRADIENT,
            self.GRAD_CAM,
            self.HIRES_CAM,
            self.ABLATION_CAM,
            self.GRAD_CAM_PLUS_PLUS,
            self.GROUP_CAM,
            self.WGRAD_CAM,
            self.XGRAD_CAM,
            self.LIME,
            self.XRAI,
            self.KERNEL_SHAP,
            self.LRP_EPSILON,
            self.LRP_GAMMA,
            self.LRP_ALPHA_BETA,
            self.LRP_ZPLUS,
            self.LRP_W2,
        ]

        self.ALL_DIRS = [
            self.MASK_RISE,
            self.MASK_SLIC,
            self.MASK_SEED,
            self.ACTIVATIONS,
            self.STATISTICS,
            self.GRADIENT,
            self.GRADIENT_SQUARED,
            self.DECONVOLUTION,
            self.GUIDED_BACKPROPAGATION,
            self.INPUT_GRADIENT,
            self.INTEGRATED_GRADIENT,
            self.SMOOTH_GRADIENT,
            self.SMOOTH_GRADIENT_SQUARE,
            self.VAR_GRADIENT,
            self.GRAD_CAM,
            self.GRAD_CAM_PLUS_PLUS,
            self.HIRES_CAM,
            self.WGRAD_CAM,
            self.ABLATION_CAM,
            self.GROUP_CAM,
            self.XGRAD_CAM,
            self.LRP_EPSILON,
            self.LRP_GAMMA,
            self.LRP_ALPHA_BETA,
            self.LRP_ZPLUS,
            self.LRP_W2,
            self.XRAI,
            self.LIME,
            self.KERNEL_SHAP,
        ]

        self.STATISTICS_TYPES = [
            self.INSERTION,
            self.DELETION,
            self.OVERALL,
            self.RATE_OF_INCREASE_IN_SCORES,
            self.AVERAGE_DROP,
        ]

        if self.GRADIENT in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GRADIENT] = "Gradient"
        if self.GRADIENT_SQUARED in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GRADIENT_SQUARED] = "Gradient Squared"
        if self.DECONVOLUTION in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.DECONVOLUTION] = "Deconvolution"
        if self.GUIDED_BACKPROPAGATION in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GUIDED_BACKPROPAGATION] = "Guided Backpropagation"
        if self.INPUT_GRADIENT in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.INPUT_GRADIENT] = "Input Gradient"

        if self.INTEGRATED_GRADIENT in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.INTEGRATED_GRADIENT] = "Integrated Gradient"

        if self.SMOOTH_GRADIENT in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.SMOOTH_GRADIENT] = "Smooth Gradient"
        if self.SMOOTH_GRADIENT_SQUARE in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.SMOOTH_GRADIENT_SQUARE] = "Smooth Gradient Squared"
        if self.VAR_GRADIENT in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.VAR_GRADIENT] = "Var Gradient"

        if self.GRAD_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GRAD_CAM] = "Grad-CAM"
        if self.GRAD_CAM_PLUS_PLUS in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GRAD_CAM_PLUS_PLUS] = "Grad-CAM++"
        if self.WGRAD_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.WGRAD_CAM] = "Weighted Grad-CAM"
        if self.HIRES_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.HIRES_CAM] = "HiRes-CAM"
        if self.ABLATION_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.ABLATION_CAM] = "Ablation-CAM"
        if self.GROUP_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.GROUP_CAM] = "Group-CAM"
        if self.XGRAD_CAM in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.XGRAD_CAM] = "XGrad-CAM"

        if self.LIME in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LIME] = "LIME"
        if self.XRAI in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.XRAI] = "XRAI"
        if self.KERNEL_SHAP in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.KERNEL_SHAP] = "Kernel SHAP"

        if self.LRP_EPSILON in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LRP_EPSILON] = "LRP Epsilon"
        if self.LRP_GAMMA in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LRP_GAMMA] = "LRP Gamma"
        if self.LRP_ALPHA_BETA in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LRP_ALPHA_BETA] = "LRP Alpha-Beta"
        if self.LRP_ZPLUS in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LRP_ZPLUS] = "LRP ZPlus"
        if self.LRP_W2 in self.AVAILABLE_HEATMAPS:
            self.XAI_TRANSLATIONS[self.LRP_W2] = "LRP W2"

    CMAP: str = "hot"
    BASELINE: int = 0
    STEPS: int = 50
    NOISE_LEVEL: float = 0.1
    NUM_SAMPLES: int = 50
    CACHE_ENABLED: bool = False
    BATCH_SIZE: int = 16

    EPSILON: float = 1e-8
    AGGREGATION: str = "max"  # mean, sum, max

    DEFAULT_EPSILON: float = 1e-6
    DEFAULT_AGGREGATION: str = "max"
    DEFAULT_N_SAMPLES: int = 50
    DEFAULT_NOISE_LEVEL: float = 0.1
    DEFAULT_GROUP_SIZE: int = 8
    DEFAULT_KSIZE: int = 3
    DEFAULT_SIGMA: float = 0.5
    DEFAULT_BASELINE: int = 0
    DEFAULT_STEPS: int = 50
    DEFAULT_SEGMENTATION: str = "seed"
    DEFAULT_GAMMA: float = 0.75
    DEFAULT_ALPHA: float = 2.5
    DEFAULT_BETA: float = 1.5
    DEFAULT_SUPER_PIXELS: int = 30

    CONFIG_FILENAME: str = "config.json"

    THRESHOLD_ACTIVATIONS: float = 0.1

    SEEDS_ITERATIONS: int = 5
    SLIC_ITERATIONS: int = 5
    SEEDS_LEVELS: int = 2
