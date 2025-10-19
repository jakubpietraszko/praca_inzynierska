from fastapi import APIRouter, Depends

from interfaces.all_models import AllModelsResponse
from interfaces.model_layers import ModelLayersRequest, ModelLayersResponse
from services.container import get_models_service

router = APIRouter()


@router.get("/all-models", response_model=AllModelsResponse)
async def all_models(models_service=Depends(get_models_service)):  # noqa: B008
    models = await models_service.list_available_models()
    return AllModelsResponse(models=models)


@router.post("/model-layers", response_model=ModelLayersResponse)
async def model_layers(request: ModelLayersRequest, models_service=Depends(get_models_service)):  # noqa: B008
    layers = await models_service.get_model_layers(request.model)
    return ModelLayersResponse(layers=layers)


@router.get("/model-layers", response_model=ModelLayersResponse)
async def model_layers_get(model: str, models_service=Depends(get_models_service)):  # noqa: B008
    layers = await models_service.get_model_layers(model)
    return ModelLayersResponse(layers=layers)
