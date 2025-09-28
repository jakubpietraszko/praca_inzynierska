import numpy as np
import os
from typing import Tuple
from interfaces.all_heatmaps import AllHeatmapsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from interfaces.label_filename import LabelFilenameTarget
from interfaces.unsupervised_heatmaps import UnsupervisedHeatmapsResponse
from interfaces.unsupervised_heatmaps import FilenameCoords

from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler


class HeatmapsService:
    def __init__(self,
                 dataset_path: str,
                 models_datasets_path: str):
        from services.datasets_service import DatasetsService
        from services.container import get_datasets_service
        from services.heatmaps_facede_service import HeatmapsFacedeService
        from services.container import get_heatmaps_facede_service
        from services.container import get_container
        from services.utilities_service import UtilitiesService
        from services.container import get_utilities_service

        self.dataset_service: DatasetsService = get_datasets_service()
        self.heatmaps_facede_service: HeatmapsFacedeService = get_heatmaps_facede_service()
        self.container = get_container()
        self.utilities_service: UtilitiesService = get_utilities_service()

        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path

    async def activations(self) -> Tuple[np.ndarray, int]:
        pass

    async def all_available_heatmaps(self, constants):
        heatmaps = constants.AVAILABLE_HEATMAPS
        return AllHeatmapsResponse(
            heatmaps=heatmaps
        )

    async def unsupervised_heatmaps(self, request):
        filenames = await self.dataset_service.list_filenames_in_label(
            dataset_name=request.dataset,
            label=request.label
        )

        req = GradientRequest(
            model=request.model,
            dataset=request.dataset,
            images=[
                LabelFilenameTarget(
                    label=request.label,
                    filename=filename,
                    target=None
                ) for filename in filenames
            ],
            overwrite=False
        )

        await self.heatmaps_facede_service.generate_heatmap(
            request=req,
            type=request.heatmap
        )

        images_dir = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            request.heatmap,
            request.label
        )

        image_paths = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
        ]

        if len(image_paths) == 0:
            return UnsupervisedHeatmapsResponse(
                dataset=request.dataset,
                label_filenames_coords=[],
                success=False
            )

        images = list()

        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)

        images_np = np.array(images)
        n_samples = images_np.shape[0]
        images_flat_np = images_np.reshape(n_samples, -1)
        tsne = TSNE(n_components=2)
        images_tsne = tsne.fit_transform(images_flat_np)
        images_tsne = normalize(images_tsne, axis=0)
        scaler = MinMaxScaler()
        images_tsne = scaler.fit_transform(images_tsne)
        filenames_coords = list()

        for image_path, coords in zip(image_paths, images_tsne):
            filenames_coords.append(
                FilenameCoords(
                    filename=os.path.basename(image_path),
                    coordX=float(coords[0]),
                    coordY=float(coords[1])
                )
            )

        return UnsupervisedHeatmapsResponse(
            filenames_coords=filenames_coords,
            success=True
        )
