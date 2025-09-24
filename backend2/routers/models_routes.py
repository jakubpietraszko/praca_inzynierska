from fastapi import APIRouter, Depends

from services.container import get_models_service
from interfaces.all_models import AllModelsResponse
from interfaces.model_layers import ModelLayersRequest, ModelLayersResponse

router = APIRouter()

@router.get('/all-models', response_model=AllModelsResponse)
async def all_models(models_service=Depends(get_models_service)):
    models = await models_service.list_available_models()
    return AllModelsResponse(models=models)

@router.post('/model-layers', response_model=ModelLayersResponse)
async def model_layers(
    request: ModelLayersRequest,
    models_service=Depends(get_models_service)
):
    layers = await models_service.get_model_layers(request.model)
    return ModelLayersResponse(layers=layers)



