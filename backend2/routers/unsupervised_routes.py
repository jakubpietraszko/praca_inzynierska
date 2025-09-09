from fastapi import APIRouter, Depends

from constans import Constants
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
from interfaces.unsupervised_activations import LabelFilenameCoords, UnsupervisedActivationsRequest, UnsupervisedActivationsResponse
from interfaces.unsupervised_heatmaps import UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
import numpy as np
import os
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler

router = APIRouter()


@router.post('/activations', response_model=UnsupervisedActivationsResponse)
async def activations(request: UnsupervisedActivationsRequest,
                      container: ServiceContainer = Depends(get_container),
                      models_service: ModelsService = Depends(
                          get_models_service),
                      heatmaps_service: HeatmapsService = Depends(
                          get_heatmaps_service),
                      utilities_service: UtilitiesService = Depends(
                          get_utilities_service),
                      models_datasets_service: ModelsDatasetsService = Depends(
                          get_models_datasets_service),
                      datasets_service: DatasetsService = Depends(
                          get_datasets_service),
                      constants: Constants = Depends(get_constants)):
    images_dir = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ACTIVATIONS,
        request.label,
        utilities_service.delete_extension(request.filename)
    )
    image_paths = [
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
    ]
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

    label_filenames_coords = list()

    for image_path, coords in zip(image_paths, images_tsne):
        label_filenames_coords.append(
            LabelFilenameCoords(
                label=request.label,
                filename=os.path.basename(image_path),
                coordX=float(coords[0]),
                coordY=float(coords[1])
            )
        )

    return UnsupervisedActivationsResponse(
        dataset=request.dataset,
        label_filenames_coords=label_filenames_coords,
        success=True
    )



@router.post('/heatmaps', response_model=UnsupervisedHeatmapsResponse)
async def heatmaps(request: UnsupervisedHeatmapsRequest,
                   container: ServiceContainer = Depends(get_container),
                   models_service: ModelsService = Depends(
                       get_models_service),
                   heatmaps_service: HeatmapsService = Depends(
                       get_heatmaps_service),
                   utilities_service: UtilitiesService = Depends(
                       get_utilities_service),
                   models_datasets_service: ModelsDatasetsService = Depends(
                       get_models_datasets_service),
                   datasets_service: DatasetsService = Depends(
                       get_datasets_service),
                   constants: Constants = Depends(get_constants)):
    pass