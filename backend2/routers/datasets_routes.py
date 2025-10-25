from fastapi import APIRouter, Depends

from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_images import AllImagesResponse, AllsImagesRequest
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.datasets_balance import DatasetBalanceResponse
from interfaces.fetch_image import (
    FetchActivationImageRequest,
    FetchImageRequest,
    FetchImageResponse,
    FetchModelsDatasetsRequest,
)
from services.container import (
    Constants,
    get_constants,
    get_datasets_service,
    get_models_datasets_service,
    get_utilities_service,
)
from services.datasets_service import DatasetsService
from services.models_datasets_service import ModelsDatasetsService
from services.utilities_service import UtilitiesService

router = APIRouter()


@router.get("/all-datasets", response_model=AllDatasetsResponse)
async def all_datasets(datasets_service: DatasetsService = Depends(get_datasets_service)):  # noqa: B008
    datasets = await datasets_service.list_available_datasets()
    return AllDatasetsResponse(datasets=datasets)


@router.get("/all-labels/{dataset}", response_model=AllLabelsResponse)
async def all_labels_get(
    dataset: str,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    labels = await datasets_service.list_labels_in_dataset(dataset)
    return AllLabelsResponse(labels=labels)


@router.post("/all-labels", response_model=AllLabelsResponse)
async def all_labels(
    request: AllLabelsRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    labels = await datasets_service.list_labels_in_dataset(request.dataset)
    return AllLabelsResponse(labels=labels)


@router.post("/all-images", response_model=AllImagesResponse)
async def all_images(
    request: AllsImagesRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    images = await datasets_service.list_filenames_in_label(request.dataset, request.label)
    return AllImagesResponse(images=images)


@router.get("/all-images", response_model=AllImagesResponse)
async def all_images_get(
    dataset: str,
    label: str,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    try:
        images = await datasets_service.list_filenames_in_label(dataset, label)
    except:
        images = []
    return AllImagesResponse(images=images)


@router.post("/fetch-image", response_model=FetchImageResponse)
async def fetch_image_post(
    request: FetchImageRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    image = await datasets_service.fetch_image(dataset=request.dataset, label=request.label, filename=request.filename)
    return FetchImageResponse(image=image)


@router.get("/fetch-image", response_model=FetchImageResponse)
async def fetch_image_get(
    dataset: str,
    label: str,
    filename: str,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    image = await datasets_service.fetch_image(dataset=dataset, label=label, filename=filename)
    return FetchImageResponse(image=image)


@router.post("/fetch-activations-image", response_model=FetchImageResponse)
async def fetch_activations_image_post(
    request: FetchActivationImageRequest,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
    constants: Constants = Depends(get_constants),  # noqa: B008
):
    image = await models_data_service.fetch_activations_image(
        await utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ACTIVATIONS,
        request.label,
        await utilities_service.delete_extension(request.filename),
        request.image,
    )

    return FetchImageResponse(image=image)


@router.get("/fetch-activations-image", response_model=FetchImageResponse)
async def fetch_activations_image_get(
    dataset: str,
    model: str,
    label: str,
    filename: str,
    image: str,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
    constants: Constants = Depends(get_constants),  # noqa: B008
):
    image_ = await models_data_service.fetch_activations_image(
        await utilities_service.models_datasets_path(model, dataset),
        constants.ACTIVATIONS,
        label,
        await utilities_service.delete_extension(filename),
        image,
    )

    return FetchImageResponse(image=image_)


@router.get("/fetch-any-activations-image", response_model=FetchImageResponse)
async def fetch_any_activations_image_get(
    dataset: str,
    model: str,
    label: str,
    filename: str,
    image: str,
    ref: str,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
    constants: Constants = Depends(get_constants),  # noqa: B008
):
    image_ = await models_data_service.fetch_any_activations_image(
        await utilities_service.models_datasets_path(model, dataset),
        constants.ACTIVATIONS,
        label,
        await utilities_service.delete_extension(filename),
        image,
        ref,
    )

    return FetchImageResponse(image=image_)


@router.post("/fetch-models-datasets-image", response_model=FetchImageResponse)
async def fetch_models_datasets_image_post(
    request: FetchModelsDatasetsRequest,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
):
    image = await models_data_service.fetch_image(
        await utilities_service.models_datasets_path(request.model, request.dataset),
        request.type,
        request.label,
        request.filename,
    )

    return FetchImageResponse(image=image)


@router.get("/fetch-models-datasets-image", response_model=FetchImageResponse)
async def fetch_models_datasets_image_get(
    model: str,
    dataset: str,
    type: str,
    label: str,
    filename: str,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
):
    image = await models_data_service.fetch_image(
        await utilities_service.models_datasets_path(model, dataset), type, label, filename
    )

    return FetchImageResponse(image=image)


@router.get("/fetch-dataset-fourier-image", response_model=FetchImageResponse)
async def fetch_dataset_fourier_image(
    dataset: str,
    label: str,
    filename: str,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    image = await datasets_service.fetch_fourier_image(dataset=dataset, label=label, filename=filename)
    return FetchImageResponse(image=image)


@router.get("/fetch-models-datasets-fourier-image", response_model=FetchImageResponse)
async def fetch_models_datasets_fourier_image(
    model: str,
    dataset: str,
    type: str,
    label: str,
    filename: str,
    models_data_service: ModelsDatasetsService = Depends(get_models_datasets_service),  # noqa: B008
    utilities_service: UtilitiesService = Depends(get_utilities_service),  # noqa: B008
):
    image = await models_data_service.fetch_fourier_image(
        await utilities_service.models_datasets_path(model, dataset), type, label, filename
    )

    return FetchImageResponse(image=image)


@router.get("/datasets-balance", response_model=DatasetBalanceResponse)
async def dataset_balance(
    dataset: str,
    datasets_service: DatasetsService = Depends(get_datasets_service),  # noqa: B008
):
    return await datasets_service.datasets_balance(dataset)
