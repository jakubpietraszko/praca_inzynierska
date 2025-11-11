import os

import numpy as np
from PIL import Image

from interfaces.heatmap import HeatmapRequest
from interfaces.label_filename import LabelFilenameTarget
from interfaces.unsupervised_activations import (
    LabelFilenameCoords,
    UnsupervisedActivationsRequest,
    UnsupervisedActivationsResponse,
)
from interfaces.unsupervised_dataset import (
    UnsupervisedDatasetRequest,
    UnsupervisedDatasetResponse,
)
from interfaces.unsupervised_heatmaps import FilenameCoords, UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
from interfaces.unsupervised_label import UnsupervisedLabelRequest, UnsupervisedLabelResponse
from utils.unsupervised import apply_tsne_with_augmentation, generate_colors


class UnsupervisedService:
    def __init__(self) -> None:
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
        images_dir: str = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.ACTIVATIONS,
            request.label,
            await self.utilities_service.delete_extension(request.filename),
        )
        image_paths: list[str] = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        images = []

        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)

        images_np = np.array(images)
        n_samples = images_np.shape[0]
        images_flat_np = images_np.reshape(n_samples, -1)

        images_tsne = await apply_tsne_with_augmentation(images_flat_np)

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

        req = HeatmapRequest(
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

        images_tsne = await apply_tsne_with_augmentation(images_flat_np)

        filenames_coords = []

        for image_path, coords in zip(image_paths, images_tsne, strict=False):
            filenames_coords.append(
                FilenameCoords(filename=os.path.basename(image_path), coordX=float(coords[0]), coordY=float(coords[1]))
            )

        return UnsupervisedHeatmapsResponse(filenames_coords=filenames_coords, success=True)

    async def any_activations(self) -> None:
        pass

    async def dataset(self, request: UnsupervisedDatasetRequest) -> UnsupervisedDatasetResponse:
        from interfaces.unsupervised_dataset import Color, Coords, UnsupervisedDatasetItem

        cached_data = await self.datasets_service.load_unsupervised_cache(request.dataset)

        if cached_data is not None:
            items = [
                UnsupervisedDatasetItem(
                    filename=item["filename"],
                    label=item["label"],
                    coords=Coords(x=item["coords"]["x"], y=item["coords"]["y"]),
                    color=Color(r=item["color"]["r"], g=item["color"]["g"], b=item["color"]["b"]),
                )
                for item in cached_data["items"]
            ]
            return UnsupervisedDatasetResponse(items=items, success=True)

        labels = await self.datasets_service.list_labels_in_dataset(request.dataset)

        all_images = []
        all_filenames = []
        all_labels = []

        for label in labels:
            filenames = await self.datasets_service.list_filenames_in_label(request.dataset, label)

            for filename in filenames:
                image_path = os.path.join(self.datasets_service.datasets_path, request.dataset, label, filename)
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                all_images.append(img_array.flatten())
                all_filenames.append(filename)
                all_labels.append(label)

        images_flat_np = np.array(all_images)

        images_tsne = await apply_tsne_with_augmentation(images_flat_np)

        colors = generate_colors(len(labels))
        label_to_color = {}

        for idx, label in enumerate(labels):
            label_to_color[label] = colors[idx]

        items = []
        for filename, label, coords in zip(all_filenames, all_labels, images_tsne, strict=False):
            color = label_to_color[label]
            items.append(
                UnsupervisedDatasetItem(
                    filename=filename,
                    label=label,
                    coords=Coords(x=float(coords[0]), y=float(coords[1])),
                    color=Color(r=color[0], g=color[1], b=color[2]),
                )
            )

        await self.datasets_service.save_unsupervised_cache(
            request.dataset,
            {
                "items": [
                    {
                        "filename": item.filename,
                        "label": item.label,
                        "coords": {"x": item.coords.x, "y": item.coords.y},
                        "color": {"r": item.color.r, "g": item.color.g, "b": item.color.b},
                    }
                    for item in items
                ]
            },
        )

        return UnsupervisedDatasetResponse(items=items, success=True)

    async def label(self, request: UnsupervisedLabelRequest) -> UnsupervisedLabelResponse:
        from interfaces.unsupervised_label import Color, Coords, UnsupervisedLabelItem

        cached_data = await self.datasets_service.load_unsupervised_label_cache(request.dataset, request.label)

        if cached_data is not None:
            items = [
                UnsupervisedLabelItem(
                    filename=item["filename"],
                    coords=Coords(x=item["coords"]["x"], y=item["coords"]["y"]),
                    color=Color(r=item["color"]["r"], g=item["color"]["g"], b=item["color"]["b"]),
                )
                for item in cached_data["items"]
            ]
            return UnsupervisedLabelResponse(items=items, success=True)

        filenames = await self.datasets_service.list_filenames_in_label(request.dataset, request.label)

        all_images = []
        all_filenames = []

        for filename in filenames:
            image_path = os.path.join(self.datasets_service.datasets_path, request.dataset, request.label, filename)
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            all_images.append(img_array.flatten())
            all_filenames.append(filename)

        images_flat_np = np.array(all_images)

        images_tsne = await apply_tsne_with_augmentation(images_flat_np)

        colors = generate_colors(1)
        color = colors[0]

        items = []
        for filename, coords in zip(all_filenames, images_tsne, strict=False):
            items.append(
                UnsupervisedLabelItem(
                    filename=filename,
                    coords=Coords(x=float(coords[0]), y=float(coords[1])),
                    color=Color(r=color[0], g=color[1], b=color[2]),
                )
            )

        await self.datasets_service.save_unsupervised_label_cache(
            request.dataset,
            request.label,
            {
                "items": [
                    {
                        "filename": item.filename,
                        "coords": {"x": item.coords.x, "y": item.coords.y},
                        "color": {"r": item.color.r, "g": item.color.g, "b": item.color.b},
                    }
                    for item in items
                ]
            },
        )

        return UnsupervisedLabelResponse(items=items, success=True)
