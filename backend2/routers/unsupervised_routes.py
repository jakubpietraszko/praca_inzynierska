import os

import numpy as np
from fastapi import APIRouter, Depends
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, normalize

from constans import Constants
from interfaces.gradient import GradientRequest
from interfaces.label_filename import LabelFilenameTarget
from interfaces.unsupervised_activations import (
    LabelFilenameCoords,
    UnsupervisedActivationsRequest,
    UnsupervisedActivationsResponse,
)
from interfaces.unsupervised_heatmaps import FilenameCoords, UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
from services.container import (
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

router = APIRouter()


@router.post("/activations", response_model=UnsupervisedActivationsResponse)
async def activations(request: UnsupervisedActivationsRequest):
    container: ServiceContainer = Depends(get_container)
    utilities_service: UtilitiesService = Depends(get_utilities_service)
    constants: Constants = Depends(get_constants)

    images_dir = os.path.join(
        container.models_datasets_path,
        await utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ACTIVATIONS,
        request.label,
        await utilities_service.delete_extension(request.filename),
    )
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    images = []

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


@router.post("/heatmaps", response_model=UnsupervisedHeatmapsResponse)
async def heatmaps(request: UnsupervisedHeatmapsRequest):
    container: ServiceContainer = Depends(get_container)
    utilities_service: UtilitiesService = Depends(get_utilities_service)
    datasets_service: DatasetsService = Depends(get_datasets_service)
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service)

    filenames = await datasets_service.list_filenames_in_label(dataset_name=request.dataset, label=request.label)

    req = GradientRequest(
        model=request.model,
        dataset=request.dataset,
        images=[LabelFilenameTarget(label=request.label, filename=filename, target=None) for filename in filenames],
        overwrite=False,
    )

    await heatmaps_facede_service.generate_heatmap(request=req, type=request.heatmap)

    images_dir = os.path.join(
        container.models_datasets_path,
        await utilities_service.models_datasets_path(request.model, request.dataset),
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
    tsne = TSNE(n_components=2)
    images_tsne = tsne.fit_transform(images_flat_np)

    images_tsne = normalize(images_tsne, axis=0)

    scaler = MinMaxScaler()
    images_tsne = scaler.fit_transform(images_tsne)

    filenames_coords = []

    for image_path, coords in zip(image_paths, images_tsne, strict=False):
        filenames_coords.append(
            FilenameCoords(filename=os.path.basename(image_path), coordX=float(coords[0]), coordY=float(coords[1]))
        )

    return UnsupervisedHeatmapsResponse(filenames_coords=filenames_coords, success=True)
