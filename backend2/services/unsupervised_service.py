import os

import numpy as np
from PIL import Image

from interfaces.gradient import GradientRequest
from interfaces.label_filename import LabelFilenameTarget
from interfaces.unsupervised_activations import (
    LabelFilenameCoords,
    UnsupervisedActivationsRequest,
    UnsupervisedActivationsResponse,
)
from interfaces.unsupervised_heatmaps import FilenameCoords, UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
from utils.unsupervised import apply_tsne_with_augmentation


class UnsupervisedService:
    def __init__(self):
        from services.container import (
            Constants,
            DatasetsService,
            HeatmapsFacedeService,
            ServiceContainer,
            UtilitiesService,
            get_constants,
            get_container,
            get_datasets_service,
            get_heatmaps_facede_service,
            get_utilities_service,
        )

        self.constants: Constants = get_constants()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.container: ServiceContainer = get_container()
        self.heatmaps_facede_service: HeatmapsFacedeService = get_heatmaps_facede_service()

    async def activations(self, request: UnsupervisedActivationsRequest) -> UnsupervisedActivationsResponse:
        images_dir = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.ACTIVATIONS,
            request.label,
            await self.utilities_service.delete_extension(request.filename),
        )
        image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        images = []

        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)

        images_np = np.array(images)
        n_samples = images_np.shape[0]
        images_flat_np = images_np.reshape(n_samples, -1)

        images_tsne = apply_tsne_with_augmentation(images_flat_np)

        label_filenames_coords = []

        for image_path, coords in zip(image_paths, images_tsne, strict=False):
            label_filenames_coords.append(
                LabelFilenameCoords(
                    label=request.label,
                    filename=os.path.basename(image_path),
                    coordX=float(coords[0]),
                    coordY=float(coords[1]),
                )
            )

        return UnsupervisedActivationsResponse(
            dataset=request.dataset, label_filenames_coords=label_filenames_coords, success=True
        )

    async def heatmaps(self, request: UnsupervisedHeatmapsRequest) -> UnsupervisedHeatmapsResponse:
        filenames = await self.datasets_service.list_filenames_in_label(
            dataset_name=request.dataset, label=request.label
        )

        req = GradientRequest(
            model=request.model,
            dataset=request.dataset,
            images=[LabelFilenameTarget(label=request.label, filename=filename, target=None) for filename in filenames],
            overwrite=False,
        )

        await self.heatmaps_facede_service.generate_heatmap(request=req, type=request.heatmap)

        images_dir = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            request.heatmap,
            request.label,
        )

        image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]

        if len(image_paths) == 0:
            return UnsupervisedHeatmapsResponse(filenames_coords=[], success=False)

        images = []

        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)

        images_np = np.array(images)
        n_samples = images_np.shape[0]
        images_flat_np = images_np.reshape(n_samples, -1)

        images_tsne = apply_tsne_with_augmentation(images_flat_np)

        filenames_coords = []

        for image_path, coords in zip(image_paths, images_tsne, strict=False):
            filenames_coords.append(
                FilenameCoords(filename=os.path.basename(image_path), coordX=float(coords[0]), coordY=float(coords[1]))
            )

        return UnsupervisedHeatmapsResponse(filenames_coords=filenames_coords, success=True)

    async def any_activations(self):
        pass
