from fastapi import APIRouter, Depends
from services.container import get_models_datasets_service
from interfaces.init_dir import InitDirRequest, InitDirResponse
from services.models_datasets_service import ModelsDatasetsService

router = APIRouter()

@router.post('/init-dir', response_model=InitDirResponse)
async def init_dir(request: InitDirRequest, models_datasets_service: ModelsDatasetsService = Depends(get_models_datasets_service)):
    success = models_datasets_service.init_dir(request.model, request.dataset)
    return InitDirResponse(success=success)

