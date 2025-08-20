import os
from functools import lru_cache
from .model_service import ModelService
from .dataset_service import DatasetService
from .mask_service import MaskService


class ServiceContainer:
    def __init__(self, env: str = 'development'):
        self.env = env
        
        if env == 'development':
            self.base_dir = os.path.join('..', 'DIR')
        else:
            self.base_dir = os.path.join('/app', 'DIR')
        
        self.datasets_path = os.path.join(self.base_dir, 'datasets')
        self.models_path = os.path.join(self.base_dir, 'models')
        self.models_datasets_path = os.path.join(self.base_dir, 'model-dataset')
        
        self._model_service = None
        self._dataset_service = None
        self._mask_service = None
    
    @property
    def model_service(self) -> ModelService:
        if self._model_service is None:
            self._model_service = ModelService(self.models_path)
        return self._model_service
    
    @property
    def dataset_service(self) -> DatasetService:
        if self._dataset_service is None:
            self._dataset_service = DatasetService(self.datasets_path)
        return self._dataset_service
    
    @property
    def mask_service(self) -> MaskService:
        if self._mask_service is None:
            self._mask_service = MaskService(self.models_datasets_path)
        return self._mask_service


_container: ServiceContainer = None


def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        env = os.getenv('APP_ENV', 'development')
        _container = ServiceContainer(env)
    return _container


def get_model_service() -> ModelService:
    return get_container().model_service


def get_dataset_service() -> DatasetService:
    return get_container().dataset_service


def get_mask_service() -> MaskService:
    return get_container().mask_service
