import os

import cv2
import numpy as np
import torch


class UtilitiesService:
    def __init__(self):
        pass

    async def env(self):
        env = os.getenv('APP_ENV', 'development')
        return env

    async def models_datasets_path(self, model: str, dataset: str) -> str:
        result = model + '_' + dataset
        return result

    async def convert_image_to_tensor(self, image: np.ndarray, grad: bool = False):
        assert len(image.shape) == 3
        assert image.dtype == np.uint8

        image = image.astype(np.float32)
        image = image / 255.0

        image_tensor: torch.Tensor = torch.from_numpy(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.float()

        if grad:
            image_tensor.requires_grad = True

        return image_tensor

    async def delete_extension(self, filename: str) -> str:
        base = filename.split('.')[0]
        return base

    async def add_png_extension(self, filename: str) -> str:
        base = await self.delete_extension(filename)
        base += '.png'
        return base

    async def relu_normalize(self, image: np.ndarray, shape: tuple, epsilon: float) -> np.ndarray:
        assert len(image.shape) == 2

        image = np.maximum(image, 0)
        image = (image - image.min()) / (image.max() - image.min() + epsilon)

        image = cv2.resize(image, (shape[2], shape[1]))
        assert len(image.shape) == 2

        image = image * 255.0
        image = image.astype(np.uint8)

        assert image.dtype == np.uint8

        return image

    async def aggregate_channels(self, image: np.ndarray, type: str) -> np.ndarray:
        assert len(image.shape) == 3

        if type == 'max':
            image = np.max(image, axis=0)
        elif type == 'mean':
            image = np.mean(image, axis=0)
        elif type == 'sum':
            image = np.sum(image, axis=0)

        assert len(image.shape) == 2

        return image

    async def skip_forward(self, path) -> bool:
        return bool(os.path.exists(path))

    async def fill_target(self, model_info, label) -> int:
        return model_info.label_to_idx[label]
