import os
import base64
from typing import Dict
import numpy as np
from PIL import Image


class ModelsDatasetsService:
    def __init__(self, models_datasets_path: str, datasets_path: str, utilities_service, constants):
        from services.container import get_constants

        from services.container import get_datasets_service
        from services.datasets_service import DatasetsService

        from services.utilities_service import UtilitiesService

        self.models_datasets_path = models_datasets_path
        self.datasets_path = datasets_path

        self.utilities_service: UtilitiesService = utilities_service

        self.constants = get_constants()
        self.datasets_service: DatasetsService = get_datasets_service()

    async def init_dir(self, model: str, dataset: str) -> bool:
        try:
            await self.full_init_dir(model, dataset)
        except:
            return False
        return True

    async def load_image(self,
                         model: str,
                         dataset: str,
                         label: str,
                         heatmap: str,
                         filename: str) -> np.ndarray:

        path = os.path.join(
            self.models_datasets_path,
            await self.utilities_service.models_datasets_path(model, dataset),
            heatmap,
            label,
            filename
        )

        image = Image.open(path).convert('L')
        image = np.array(image)

        return image

    async def fetch_activations_image(self,
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

    async def fetch_image(self,
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

    async def full_init_dir(self, model: str, dataset: str) -> bool:
        name_of_dir = await self.utilities_service.models_datasets_path(
            model, dataset)
        full_path = os.path.join(self.models_datasets_path, name_of_dir)

        try:
            os.makedirs(full_path, exist_ok=True)

            path = os.path.join(full_path, self.constants.ACTIVATIONS)
            os.makedirs(path, exist_ok=True)

            for label in await self.datasets_service.list_labels_in_dataset(dataset):
                path = os.path.join(
                    full_path, self.constants.ACTIVATIONS, label)
                os.makedirs(path, exist_ok=True)

                for filename in await self.datasets_service.list_filenames_in_label(dataset, label):
                    path = os.path.join(
                        full_path, self.constants.ACTIVATIONS, label, await self.utilities_service.delete_extension(
                            filename
                        ))
                    os.makedirs(path, exist_ok=True)

            for dir_name in self.constants.AVAILABLE_HEATMAPS:
                path = os.path.join(full_path, dir_name)
                os.makedirs(path, exist_ok=True)

                for label in await self.datasets_service.list_labels_in_dataset(dataset):
                    path = os.path.join(
                        full_path, dir_name, label)
                    os.makedirs(path, exist_ok=True)

            for dir_name in self.constants.AVAILABLE_HEATMAPS:
                path = os.path.join(
                    full_path, self.constants.STATISTICS, dir_name)
                os.makedirs(path, exist_ok=True)

                for label in await self.datasets_service.list_labels_in_dataset(dataset):
                    path = os.path.join(
                        full_path, self.constants.STATISTICS, dir_name, label)
                    os.makedirs(path, exist_ok=True)

        except:
            return False
        return True

    async def numerator_denominator(self, model: str, dataset: str, label: str) -> tuple[Dict[str, int], int]:
        dataset_dir_path = os.path.join(
            self.datasets_path, dataset, label
        )

        xai_base_path = os.path.join(
            self.models_datasets_path,
        )

        os.makedirs(xai_base_path, exist_ok=True)

        denominator = len(os.listdir(dataset_dir_path))

        numerator = dict()

        for xai_method in self.constants.AVAILABLE_HEATMAPS:
            xai_path = os.path.join(
                xai_base_path,
                await self.utilities_service.models_datasets_path(model, dataset),
                xai_method, label
            )
            os.makedirs(xai_path, exist_ok=True)

            numerator[xai_method] = len(os.listdir(xai_path))

        return numerator, denominator
