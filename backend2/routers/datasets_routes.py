from fastapi import APIRouter, Depends
from services.container import get_datasets_service
from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.all_images import AllsImagesRequest, AllImagesResponse
from interfaces.fetch_image import FetchImageRequest, FetchImageResponse
from services.datasets_service import DatasetsService

router = APIRouter()

@router.get('/all-datasets', response_model=AllDatasetsResponse)
async def all_datasets(datasets_service: DatasetsService = Depends(get_datasets_service)):
    datasets = datasets_service.list_available_datasets()
    return AllDatasetsResponse(datasets=datasets)

@router.post('/all-labels', response_model=AllLabelsResponse)
async def all_labels(
    request: AllLabelsRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    labels = datasets_service.list_labels_in_dataset(request.dataset)
    return AllLabelsResponse(labels=labels)

@router.post('/all-images', response_model=AllImagesResponse)
async def all_images(
    request: AllsImagesRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    images = datasets_service.list_filenames_in_label(request.dataset, request.label)
    return AllImagesResponse(images=images)

@router.post('/fetch-image', response_model=FetchImageResponse)
async def fetch_image(
    request: FetchImageRequest,
    datasets_service: DatasetsService = Depends(get_datasets_service)
):
    image = datasets_service.fetch_image(
        dataset=request.dataset,
        label=request.label,
        filename=request.filename
    )
    return FetchImageResponse(image=image)