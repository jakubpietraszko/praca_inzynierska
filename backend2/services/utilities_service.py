import os
import torch
import numpy as np


class UtilitiesService:
    def __init__(self):
        pass

    def env(self):
        env = os.getenv('APP_ENV', 'development')
        return env
    
    def models_datasets_path(self, model: str, dataset: str) -> str:
        result = model + '_' + dataset
        return result
    
    def convert_image_to_tensor(self, image: np.ndarray, grad: bool = False):
        assert len(image.shape) == 3
        assert image.dtype == np.uint8

        image = image.astype(np.float32)
        image = image / 255.0

        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.float()

        if grad:
            image.requires_grad = True

        return image
    
    def delete_extension(self, filename: str) -> str:
        base = filename.split('.')[0]
        return base
