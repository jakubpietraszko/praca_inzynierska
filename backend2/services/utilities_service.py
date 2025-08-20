import os


class UtilitiesService:
    def __init__(self):
        pass

    def env(self):
        env = os.getenv('APP_ENV', 'development')
        return env
    
    def models_datasets_path(self, model: str, dataset: str) -> str:
        result = model + '_' + dataset
        return result
