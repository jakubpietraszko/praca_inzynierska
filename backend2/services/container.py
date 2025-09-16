import os

from .datasets_service import DatasetsService
from .heatmaps_service import HeatmapsService
from .models_service import ModelsService
from .masks_service import MasksService
from .utilities_service import UtilitiesService
from .models_datasets_service import ModelsDatasetsService
from .gradient_service import GradientService
from .activations_service import ActivationsService
from .integrated_gradient_service import IntegratedGradientsService
from .cam_service import CAMService
from .probing_image_service import ProbingImageService
from .gradient_noise_service import GradientNoiseService
from .heatmaps_facede_service import HeatmapsFacedeService
from .lrp_service import LRPService
from .config_service import ConfigService

from constans import Constants

class ServiceContainer:
    def __init__(self, env: str = 'development'):
        self.env = env

        if env == 'development':
            self.base_dir = os.path.join('..', 'DIR')
        else:
            self.base_dir = os.path.join('/app', 'DIR')

        self.constants = Constants()

        self.datasets_path = os.path.join(self.base_dir, self.constants.DATASETS)
        self.models_path = os.path.join(self.base_dir, self.constants.MODELS)
        self.models_datasets_path = os.path.join(self.base_dir, self.constants.MODELS_DATASETS)

        self._datasets_service = None
        self._heatmaps_service = None
        self._models_service = None
        self._masks_service = None
        self._utilities_service = None
        self._models_datasets_service = None
        self._gradient_service = None
        self._activations_service = None
        self._integrated_gradients_service = None
        self._cam_service = None
        self._probing_image_service = None
        self._gradient_noise_service = None
        self._heatmaps_facede_service = None
        self._lrp_service = None
        self._config_service = None

    @property
    def datasets_service(self) -> DatasetsService:
        if self._datasets_service is None:
            self._datasets_service = DatasetsService(self.datasets_path)
        return self._datasets_service

    @property
    def heatmaps_service(self) -> HeatmapsService:
        if self._heatmaps_service is None:
            self._heatmaps_service = HeatmapsService(self.datasets_path, self.models_datasets_path)
        return self._heatmaps_service
    
    @property
    def models_service(self) -> ModelsService:
        if self._models_service is None:
            self._models_service = ModelsService(self.models_path)
        return self._models_service

    @property
    def masks_service(self) -> MasksService:
        if self._masks_service is None:
            self._masks_service = MasksService(self.datasets_path, self.models_datasets_path)
        return self._masks_service
    
    @property
    def utilities_service(self) -> UtilitiesService:
        if self._utilities_service is None:
            self._utilities_service = UtilitiesService()
        return self._utilities_service
    
    @property
    def models_datasets_service(self) -> ModelsDatasetsService:
        if self._models_datasets_service is None:
            self._models_datasets_service = ModelsDatasetsService(
                self.models_datasets_path, 
                self.utilities_service, 
                self.constants
            )
        return self._models_datasets_service
    
    @property
    def gradient_service(self) -> GradientService:
        if self._gradient_service is None:
            self._gradient_service = GradientService()
        return self._gradient_service
    
    @property
    def activations_service(self) -> ActivationsService:
        if self._activations_service is None:
            self._activations_service = ActivationsService()
        return self._activations_service
    
    @property
    def integrated_gradient_service(self) -> IntegratedGradientsService:
        if self._integrated_gradients_service is None:
            self._integrated_gradients_service = IntegratedGradientsService()
        return self._integrated_gradients_service
    
    @property
    def cam_service(self):
        if self._cam_service is None:
            from .cam_service import CAMService
            self._cam_service = CAMService()
        return self._cam_service
    
    @property
    def probing_image_service(self):
        if self._probing_image_service is None:
            self._probing_image_service = ProbingImageService()
        return self._probing_image_service
    
    @property
    def gradient_noise_service(self):
        if self._gradient_noise_service is None:
            self._gradient_noise_service = GradientNoiseService()
        return self._gradient_noise_service

    @property
    def heatmaps_facede_service(self):
        if self._heatmaps_facede_service is None:
            self._heatmaps_facede_service = HeatmapsFacedeService()
        return self._heatmaps_facede_service

    @property
    def lrp_service(self):
        if self._lrp_service is None:
            self._lrp_service = LRPService()
        return self._lrp_service
    
    @property
    def config_service(self):
        if self._config_service is None:
            self._config_service = ConfigService(self.base_dir)
        return self._config_service


_container: ServiceContainer = None

def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        env = os.getenv('APP_ENV', 'development')
        _container = ServiceContainer(env)
    return _container

def get_datasets_service() -> DatasetsService:
    return get_container().datasets_service

def get_heatmaps_service() -> HeatmapsService:
    return get_container().heatmaps_service

def get_models_service() -> ModelsService:
    return get_container().models_service

def get_masks_service() -> MasksService:
    return get_container().masks_service

def get_utilities_service() -> UtilitiesService:
    return get_container().utilities_service

def get_models_datasets_service() -> ModelsDatasetsService:
    return get_container().models_datasets_service

def get_constants() -> Constants:
    return get_container().constants

def get_gradient_service() -> GradientService:
    return get_container().gradient_service

def get_activations_service() -> ActivationsService:
    return get_container().activations_service

def get_integrated_gradients_service() -> IntegratedGradientsService:
    return get_container().integrated_gradient_service

def get_cam_service() -> CAMService:
    return get_container().cam_service

def get_probing_image_service() -> ProbingImageService:
    return get_container().probing_image_service

def get_gradient_noise_service() -> GradientNoiseService:
    return get_container().gradient_noise_service

def get_heatmaps_facede_service() -> HeatmapsFacedeService:
    return get_container().heatmaps_facede_service

def get_lrp_service() -> LRPService:
    return get_container().lrp_service

def get_config_service() -> ConfigService:
    return get_container().config_service