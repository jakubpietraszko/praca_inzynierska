from fastapi import APIRouter, Depends
from services.container import get_models_datasets_service
from interfaces.init_dir import InitDirRequest, InitDirResponse
from services.models_datasets_service import ModelsDatasetsService
from pydantic import BaseModel


class ModelDatasetI(BaseModel):
    model: str
    dataset: str


class ModelDatasetHeatmapI(ModelDatasetI):
    heatmap: str


class NumeratorDenominator(BaseModel):
    numerator: str
    denominator: str


class Response(BaseModel):
    success: bool


router = APIRouter()


@router.post('/init-dir', response_model=InitDirResponse)
async def init_dir(request: InitDirRequest, models_datasets_service: ModelsDatasetsService = Depends(get_models_datasets_service)):
    success = models_datasets_service.init_dir(request.model, request.dataset)
    return InitDirResponse(success=success)


@router.put('/models-datasets', response_model=Response)
async def models_datasets(request: ModelDatasetI,
                          models_datasets_service: ModelsDatasetsService = Depends(
                              get_models_datasets_service)
                          ):
    success = models_datasets_service.full_init_dir(
        request.model, request.dataset
    )
    return Response(success=success)
