import os

import cv2
import numpy as np
from PIL import Image

from interfaces.masks import MasksRequest, MasksResponse
from interfaces.super_pixels import SuperPixelsRequest, SuperPixelsResponse
from services.models_service import ModelInfo


class MasksService:
    def __init__(self,
                 dataset_path: str,
                 models_datasets_path: str):
        from services.container import (
            Constants,
            ModelsService,
            ServiceContainer,
            UtilitiesService,
            get_constants,
            get_container,
            get_models_service,
            get_utilities_service,
        )

        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path
        self.container: ServiceContainer = get_container()
        self.models_service: ModelsService = get_models_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.constants: Constants = get_constants()

    async def rise(self,
             height: int,
             width: int,
             scale: int = 10,
             number_of_random_mask: int = 25,
             threshold: float = 0.5,
             ) -> np.ndarray:
        masks = np.random.random(
            (number_of_random_mask, height // scale, width // scale))
        masks = np.where(masks > threshold, 1, 0).astype(np.uint8)

        result = np.zeros(
            (number_of_random_mask, height, width), dtype=np.uint8)

        for i in range(len(masks)):
            result[i] = cv2.resize(
                masks[i],
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

        return result

    async def rise_with_save(self,
                       request: MasksRequest
                       ):
        model_info: ModelInfo = await self.models_service.load_model(
            request.model
        )

        for label_filename in request.images:
            label = label_filename.label
            filename = label_filename.filename

            models_datasets_name = await self.utilities_service.models_datasets_path(
                request.model, request.dataset
            )

            path = os.path.join(
                self.container.models_datasets_path,
                models_datasets_name,
                self.constants.MASK_RISE,
                label,
                filename.split('.')[0]
            )

            os.makedirs(path, exist_ok=True)

            masks = await self.rise(
                height=model_info.input_shape[1],
                width=model_info.input_shape[2],
                scale=5,
                number_of_random_mask=request.count
            )

            for i, mask in enumerate(masks):
                mask_path = os.path.join(path, f'{i}.png')
                cv2.imwrite(mask_path, mask * 255)
        return MasksResponse(success=True)

    async def seed(self,
             image_np: np.ndarray,
             height: int,
             width: int,
             dim: int,
             super_pixels: int = 20
             ) -> tuple[np.ndarray, int]:
        LEVELS = 2
        ITERATIONS = 5

        if len(image_np.shape) == 3:
            resized_image = cv2.resize(image_np, (width, height))
        else:
            resized_image = cv2.resize(image_np, (width, height))

        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width=width,
            image_height=height,
            image_channels=dim,
            num_superpixels=super_pixels,
            num_levels=LEVELS
        )

        seeds.iterate(resized_image, ITERATIONS)
        labels = seeds.getLabels()
        return labels, np.unique(labels).shape[0]

    async def seed_with_save(self,
                       request: SuperPixelsRequest
                       ):
        model_info: ModelInfo = await self.models_service.load_model(
            request.model
        )

        path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(
                request.model, request.dataset
            ),
            self.constants.MASK_SEED,
        )

        os.makedirs(path, exist_ok=True)
        counts = list()

        for label_filename in request.images:
            label = label_filename.label
            filename = label_filename.filename

            label_path = os.path.join(
                path,
                label
            )

            os.makedirs(label_path, exist_ok=True)

            image_path_save = os.path.join(label_path, filename)

            image_path = os.path.join(
                self.container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = np.array(Image.open(image_path))
            dim = model_info.input_shape[0]
            height = model_info.input_shape[1]
            width = model_info.input_shape[2]

            mask, count = await self.seed(
                image_np=image_np,
                height=height,
                width=width,
                dim=dim,
                super_pixels=request.superPixels
            )

            counts.append(count)
            mask = cv2.normalize(
                mask, None, 0, 255, cv2.NORM_MINMAX
            )
            mask = mask.astype(np.uint8)
            cv2.imwrite(image_path_save, mask)
        return SuperPixelsResponse(success=True, superPixels=counts)

    async def slic(self,
             image_np: np.ndarray,
             height: int,
             width: int,
             dim: int,
             super_pixels: int = 20
             ) -> tuple[np.ndarray, int]:
        region_size = np.sqrt((height * width) / super_pixels)
        region_size = int(region_size)
        ITERATIONS = 5

        if len(image_np.shape) == 3:
            resized_image = cv2.resize(image_np, (width, height))
        else:
            resized_image = cv2.resize(image_np, (width, height))

        slic = cv2.ximgproc.createSuperpixelSLIC(
            resized_image,
            region_size=region_size,

        )
        slic.iterate(ITERATIONS)
        labels = slic.getLabels()
        return labels, np.unique(labels).shape[0]

    async def slic_with_save(self, request: SuperPixelsRequest):
        model_info: ModelInfo = await self.models_service.load_model(
            request.model
        )

        path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(
                request.model, request.dataset
            ),
            self.constants.MASK_SLIC,
        )

        os.makedirs(path, exist_ok=True)
        counts = list()

        for label_filename in request.images:
            label = label_filename.label
            filename = label_filename.filename

            label_path = os.path.join(
                path,
                label
            )

            os.makedirs(label_path, exist_ok=True)

            image_path_save = os.path.join(label_path, filename)

            image_path = os.path.join(
                self.container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = np.array(Image.open(image_path))
            dim = model_info.input_shape[0]
            height = model_info.input_shape[1]
            width = model_info.input_shape[2]
            mask, count = await self.slic(
                image_np=image_np,
                height=height,
                width=width,
                dim=dim,
                super_pixels=request.superPixels
            )
            counts.append(count)
            mask = cv2.normalize(
                mask, None, 0, 255, cv2.NORM_MINMAX
            )
            mask = mask.astype(np.uint8)
            cv2.imwrite(image_path_save, mask)
        return SuperPixelsResponse(success=True, superPixels=counts)
