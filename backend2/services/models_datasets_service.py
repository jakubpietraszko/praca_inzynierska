import os
import base64

class ModelsDatasetsService:
    def __init__(self, models_datasets_path: str, utilities_service, constants):
        self.models_datasets_path = models_datasets_path
        self.utilities_service = utilities_service
        self.constants = constants

    def init_dir(self, model: str, dataset: str) -> bool:
        name_of_dir = self.utilities_service.models_datasets_path(model, dataset)
        full_path = os.path.join(self.models_datasets_path, name_of_dir)

        try:
            os.makedirs(full_path, exist_ok=True)

            for dir_name in self.constants.ALL_DIRS:

                path = os.path.join(full_path, dir_name)
                os.makedirs(path, exist_ok=True)
        
        except:
            return False
        return True
    
    def fetch_activations_image(self,
                                models_dataset: str,
                                activations: str,
                                label: str,
                                filename: str,
                                image: str) -> bytes:
        path = os.path.join(
            self.models_datasets_path,
            models_dataset,
            activations,
            label,
            filename,
            image
        )

        with open(path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')

        return image
    
    def fetch_image(self,
                    models_dataset: str,
                    type: str,
                    label: str,
                    filename: str) -> bytes:
        path = os.path.join(
            self.models_datasets_path,
            models_dataset,
            type,
            label,
            filename
        )

        with open(path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')

        return image