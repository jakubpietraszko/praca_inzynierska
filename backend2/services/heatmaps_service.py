import os

import numpy as np
from PIL import Image

from interfaces.all_heatmaps import AllHeatmapsResponse
from interfaces.gradient import GradientRequest
from interfaces.label_filename import LabelFilenameTarget
from interfaces.unsupervised_heatmaps import FilenameCoords, UnsupervisedHeatmapsResponse
from utils.unsupervised import apply_tsne_with_augmentation


class HeatmapsService:
    def __init__(self, dataset_path: str, models_datasets_path: str):
        from services.container import (
            get_container,
            get_datasets_service,
            get_heatmaps_facede_service,
            get_utilities_service,
        )
        from services.datasets_service import DatasetsService
        from services.heatmaps_facede_service import HeatmapsFacedeService
        from services.utilities_service import UtilitiesService

        self.dataset_service: DatasetsService = get_datasets_service()
        self.heatmaps_facede_service: HeatmapsFacedeService = get_heatmaps_facede_service()
        self.container = get_container()
        self.utilities_service: UtilitiesService = get_utilities_service()

        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path

    async def all_available_heatmaps(self, constants):
        heatmaps = constants.AVAILABLE_HEATMAPS
        return AllHeatmapsResponse(heatmaps=heatmaps)

    async def unsupervised_heatmaps(self, request):
        filenames = await self.dataset_service.list_filenames_in_label(
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
