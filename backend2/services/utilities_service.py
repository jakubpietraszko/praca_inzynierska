import os
import torch
import numpy as np
import cv2


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
    
    def add_png_extension(self, filename: str) -> str:
        base = self.delete_extension(filename)
        base += '.png'
        return base

    def relu_normalize(self, image: np.ndarray, shape: tuple, epsilon: float) -> np.ndarray:
        assert len(image.shape) == 2

        image = np.maximum(image, 0)
        image = (image - image.min()) / (image.max() - image.min() + epsilon)

        image = cv2.resize(image, (shape[2], shape[1]))
        assert len(image.shape) == 2

        image = image * 255.0
        image = image.astype(np.uint8)

        assert image.dtype == np.uint8

        return image
    
    def aggregate_channels(self, image: np.ndarray, type: str) -> np.ndarray:
        assert len(image.shape) == 3

        if type == 'max':
            image = np.max(image, axis=0)
        elif type == 'mean':
            image = np.mean(image, axis=0)
        elif type == 'sum':
            image = np.sum(image, axis=0)

        assert len(image.shape) == 2

        return image
    
    def skip_forward(self, path) -> bool:
        if os.path.exists(path):
            return True
        return False

