from fastapi import APIRouter, Depends
from services.container import get_model_service, get_dataset_service
from services.model_service import ModelService
from services.dataset_service import DatasetService
from interfaces.all_models import AllModelsResponse
from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.all_images import AllsImagesRequest, AllImagesResponse
from interfaces.use_model import UseModelRequest, UseModelResponse
from interfaces.use_dataset import UseDatasetRequest, UseDatasetResponse
from interfaces.model_layers import ModelLayersRequest, ModelLayersResponse

router = APIRouter()


@router.get('/all-models', response_model=AllModelsResponse)
async def all_models(model_service: ModelService = Depends(get_model_service)):
    models = model_service.list_available_models()
    return AllModelsResponse(models=models)


@router.get('/all-datasets', response_model=AllDatasetsResponse)
async def all_datasets(dataset_service: DatasetService = Depends(get_dataset_service)):
    datasets = dataset_service.list_available_datasets()
    return AllDatasetsResponse(datasets=datasets)


@router.post('/all-labels', response_model=AllLabelsResponse)
async def all_labels(
    request: AllLabelsRequest, 
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        labels = dataset_service.get_dataset_labels(request.dataset)
        return AllLabelsResponse(labels=labels)
    except FileNotFoundError:
        return AllLabelsResponse(labels=[])


@router.post('/all-images', response_model=AllImagesResponse)
async def all_images(
    request: AllsImagesRequest,
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        images = dataset_service.get_images_in_label(request.dataset, request.label)
        return AllImagesResponse(images=images)
    except FileNotFoundError:
        return AllImagesResponse(images=[])


@router.post('/use-model', response_model=UseModelResponse)
async def use_model(
    request: UseModelRequest,
    model_service: ModelService = Depends(get_model_service)
):
    try:
        model_info = model_service.load_model(request.model)
        model_service.set_current_model(model_info)
        return UseModelResponse(success=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return UseModelResponse(success=False)


@router.post('/use-dataset', response_model=UseDatasetResponse)
async def use_dataset(
    request: UseDatasetRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
    model_service: ModelService = Depends(get_model_service)
):
    try:
        current_model = model_service.get_current_model()
        transform = current_model.transform if current_model else None
        
        dataset_info = dataset_service.load_dataset(request.dataset, transform)
        dataset_service.set_current_dataset(dataset_info)
        return UseDatasetResponse(success=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return UseDatasetResponse(success=False)


@router.post('/model-layers', response_model=ModelLayersResponse)
async def model_layers(
    request: ModelLayersRequest,
    model_service: ModelService = Depends(get_model_service)
):
    try:
        layers = model_service.get_model_layers(request.model)
        return ModelLayersResponse(layers=layers)
    except Exception as e:
        print(f"Error getting model layers: {e}")
        return ModelLayersResponse(layers=[])
