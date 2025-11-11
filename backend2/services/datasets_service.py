import base64
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder

from interfaces.datasets_balance import DatasetBalanceResponse


@dataclass
class DatasetInfo:
    name: str
    dataset: ImageFolder
    path: str


class DatasetsService:
    def __init__(self, datasets_path: str):
        self.datasets_path = datasets_path
        self.base_dir = os.path.dirname(datasets_path)

    async def get_unsupervised_cache_path(self, dataset_name: str) -> str:
        cache_dir = os.path.join(self.base_dir, "cache", "datasets", dataset_name)
        return os.path.join(cache_dir, "unsupervised_dataset.json")

    async def load_unsupervised_cache(self, dataset_name: str) -> dict | None:
        cache_file = await self.get_unsupervised_cache_path(dataset_name)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        return None

    async def save_unsupervised_cache(self, dataset_name: str, data: dict) -> None:
        cache_file = await self.get_unsupervised_cache_path(dataset_name)
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    async def get_unsupervised_label_cache_path(self, dataset_name: str, label: str) -> str:
        cache_dir = os.path.join(self.base_dir, "cache", "datasets", dataset_name, "labels")
        return os.path.join(cache_dir, f"{label}.json")

    async def load_unsupervised_label_cache(self, dataset_name: str, label: str) -> dict | None:
        cache_file = await self.get_unsupervised_label_cache_path(dataset_name, label)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        return None

    async def save_unsupervised_label_cache(self, dataset_name: str, label: str, data: dict) -> None:
        cache_file = await self.get_unsupervised_label_cache_path(dataset_name, label)
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

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

        if transform is None:
            transform = transforms.ToTensor()

        dataset = ImageFolder(path, transform=transform)

        dataset_info = DatasetInfo(name=dataset_name, dataset=dataset, path=path)

        return dataset_info

    async def fetch_image(self, dataset: str, label: str, filename: str) -> str:
        path = os.path.join(self.datasets_path, dataset, label, filename)

        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode("utf-8")

        return image

    async def fetch_fourier_image(self, dataset: str, label: str, filename: str) -> str:
        path = os.path.join(self.datasets_path, dataset, label, filename)

        image = Image.open(path)

        if image.mode != "L":
            image = image.convert("L")

        image_array = np.array(image)

        fft = np.fft.fft2(image_array)
        fft_shift = np.fft.fftshift(fft)

        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)

        magnitude_spectrum = (
            (magnitude_spectrum - magnitude_spectrum.min())
            / (magnitude_spectrum.max() - magnitude_spectrum.min())
            * 255
        ).astype(np.uint8)

        fourier_image = Image.fromarray(magnitude_spectrum)

        import io

        buffer = io.BytesIO()
        fourier_image.save(buffer, format="PNG")
        fourier_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return fourier_base64

    async def load_image(self, path: str, shape: tuple) -> np.ndarray:
        assert len(shape) == 3  # (1, H, W) or (3, H, W)
        assert shape[0] == 1 or shape[0] == 3  # grayscale or RGB

        image = np.array(Image.open(path))  # (H, W) or (H, W, 3)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        assert len(image.shape) == 3  # (H, W, 1) or (H, W, 3)

        width_height_tuple = (shape[2], shape[1])
        image = cv2.resize(image, width_height_tuple)  # (H, W, C) or (H, W)

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

    async def datasets_balance(self, dataset_name: str) -> DatasetBalanceResponse:
        path = os.path.join(self.datasets_path, dataset_name)
        labels = os.listdir(path)
        balance = {}
        for label in labels:
            label_path = os.path.join(path, label)
            filenames = os.listdir(label_path)
            balance[label] = len(filenames)
        return DatasetBalanceResponse(balance=balance)
