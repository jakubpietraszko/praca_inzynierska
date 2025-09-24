import os
from torchvision.datasets import ImageFolder
from dataclasses import dataclass
import base64
import numpy as np
from PIL import Image
import cv2


@dataclass
class DatasetInfo:
    name: str
    dataset: ImageFolder
    path: str

class DatasetsService:

    def __init__(self, datasets_path: str):
        self.datasets_path = datasets_path

    async def list_available_datasets(self) -> list[str]:
        datasets = os.listdir(self.datasets_path)
        return datasets
    
    async def list_labels_in_dataset(self, dataset_name: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name)
        labels = os.listdir(path)
        return labels
        

    async def list_filenames_in_label(self, dataset_name: str, label: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name, label)
        filenames = os.listdir(path)
        return filenames
    
    async def load_dataset(self, dataset_name: str, transform=None) -> DatasetInfo:
        path = os.path.join(self.datasets_path, dataset_name)

        dataset = ImageFolder(path, transform=transform)

        dataset_info = DatasetInfo(
            name=dataset_name,
            dataset=dataset,
            path=path
        )

        return dataset_info
    
    async def fetch_image(self,
                    dataset: str,
                    label: str,
                    filename: str) -> bytes:
        path = os.path.join(
            self.datasets_path,
            dataset,
            label,
            filename
        )

        with open(path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')

        return image

    
    async def load_image(self, path: str, shape: tuple) -> np.ndarray:
        assert len(shape) == 3 # (1, H, W) or (3, H, W)
        assert shape[0] == 1 or shape[0] == 3 # grayscale or RGB

        image = np.array(Image.open(path)) # (H, W) or (H, W, 3)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        assert len(image.shape) == 3 # (H, W, 1) or (H, W, 3)

        width_height_tuple = (shape[2], shape[1])
        image = cv2.resize(image, width_height_tuple) # (H, W, C) or (H, W)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))

        assert image.shape == shape
        assert image.dtype == np.uint8

        return image
        


