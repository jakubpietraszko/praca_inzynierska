from fastapi import APIRouter, Depends
from services.container import get_datasets_service, get_models_datasets_service
from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.all_images import AllsImagesRequest, AllImagesResponse
from interfaces.fetch_image import FetchImageRequest, FetchImageResponse, FetchActivationImageRequest, FetchModelsDatasetsRequest
from services.datasets_service import DatasetsService
from services.models_datasets_service import ModelsDatasetsService
from services.container import get_container, ServiceContainer
from services.utilities_service import UtilitiesService
from services.container import get_utilities_service
from services.container import get_constants, Constants
import os


router = APIRouter()


@router.get('/all-datasets', response_model=AllDatasetsResponse)
async def all_datasets(datasets_service: DatasetsService = Depends(get_datasets_service)):
    datasets = await datasets_service.list_available_datasets()
    return AllDatasetsResponse(datasets=datasets)


@router.get('/all-labels/{dataset}', response_model=AllLabelsResponse)
async def all_labels_get(
    dataset: str,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    labels = await datasets_service.list_labels_in_dataset(dataset)
    return AllLabelsResponse(labels=labels)


@router.post('/all-labels', response_model=AllLabelsResponse)
async def all_labels(
    request: AllLabelsRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    labels = await datasets_service.list_labels_in_dataset(request.dataset)
    return AllLabelsResponse(labels=labels)


@router.post('/all-images', response_model=AllImagesResponse)
async def all_images(
    request: AllsImagesRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    images = await datasets_service.list_filenames_in_label(
        request.dataset, request.label)
    return AllImagesResponse(images=images)


@router.get('/all-images', response_model=AllImagesResponse)
async def all_images_get(
    dataset: str,
    label: str,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    try:
        images = await datasets_service.list_filenames_in_label(
            dataset, label)
    except:
        images = []
    return AllImagesResponse(images=images)


@router.post('/fetch-image', response_model=FetchImageResponse)
async def fetch_image(
    request: FetchImageRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    image = await datasets_service.fetch_image(
        dataset=request.dataset,
        label=request.label,
        filename=request.filename
    )
    return FetchImageResponse(image=image)


@router.get('/fetch-image', response_model=FetchImageResponse)
async def fetch_image(
    dataset: str,
    label: str,
    filename: str,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    image = await datasets_service.fetch_image(
        dataset=dataset,
        label=label,
        filename=filename
    )
    return FetchImageResponse(image=image)


@router.post('/fetch-activations-image', response_model=FetchImageResponse)
async def fetch_activations_image(
    request: FetchActivationImageRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service),
    models_data_service: ModelsDatasetsService = Depends(
        get_models_datasets_service),
    utilities_service: UtilitiesService = Depends(get_utilities_service),
    constants: Constants = Depends(get_constants)
):

    image = await models_data_service.fetch_activations_image(
        await utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ACTIVATIONS,
        request.label,
        await utilities_service.delete_extension(request.filename),
        request.image
    )

    return FetchImageResponse(image=image)


@router.get('/fetch-activations-image', response_model=FetchImageResponse)
async def fetch_activations_image(
    dataset: str,
    model: str,
    label: str,
    filename: str,
    image: str,
    models_data_service: ModelsDatasetsService = Depends(
        get_models_datasets_service),
    utilities_service: UtilitiesService = Depends(get_utilities_service),
    constants: Constants = Depends(get_constants)
):

    image_ = await models_data_service.fetch_activations_image(
        await utilities_service.models_datasets_path(model, dataset),
        constants.ACTIVATIONS,
        label,
        await utilities_service.delete_extension(filename),
        image
    )

    return FetchImageResponse(image=image_)


@router.post('/fetch-models-datasets-image', response_model=FetchImageResponse)
async def fetch_models_datasets_image(
    request: FetchModelsDatasetsRequest,
    models_data_service: ModelsDatasetsService = Depends(
        get_models_datasets_service),
    utilities_service: UtilitiesService = Depends(get_utilities_service)
):

    image = await models_data_service.fetch_image(
        await utilities_service.models_datasets_path(request.model, request.dataset),
        request.type,
        request.label,
        request.filename
    )

    return FetchImageResponse(image=image)


@router.get('/fetch-models-datasets-image', response_model=FetchImageResponse)
async def fetch_models_datasets_image(
    model: str,
    dataset: str,
    type: str,
    label: str,
    filename: str,
    models_data_service: ModelsDatasetsService = Depends(
        get_models_datasets_service),
    utilities_service: UtilitiesService = Depends(get_utilities_service)
):

    image = await models_data_service.fetch_image(
        await utilities_service.models_datasets_path(model, dataset),
        type,
        label,
        filename
    )

    return FetchImageResponse(image=image)

@router.get('/fetch-dataset-fourier-image', response_model=FetchImageResponse)
async def fetch_dataset_fourier_image(
    dataset: str,
    label: str,
    filename: str,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    image = await datasets_service.fetch_fourier_image(
        dataset=dataset,
        label=label,
        filename=filename
    )
    return FetchImageResponse(image=image)

@router.get('/fetch-models-datasets-fourier-image', response_model=FetchImageResponse)
async def fetch_models_datasets_fourier_image(
    model: str,
    dataset: str,
    type: str,
    label: str,
    filename: str,
    models_data_service: ModelsDatasetsService = Depends(
        get_models_datasets_service),
    utilities_service: UtilitiesService = Depends(get_utilities_service)
):

    image = await models_data_service.fetch_fourier_image(
        await utilities_service.models_datasets_path(model, dataset),
        type,
        label,
        filename
    )

    return FetchImageResponse(image=image)