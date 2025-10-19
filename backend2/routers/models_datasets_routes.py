
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from interfaces.init_dir import InitDirRequest, InitDirResponse
from services.container import get_models_datasets_service
from services.models_datasets_service import ModelsDatasetsService


class ModelDatasetI(BaseModel):
    model: str
    dataset: str


class ModelDatasetLabelHeatmapI(ModelDatasetI):
    label: str
    heatmap: str

class NumeratorDenominator(BaseModel):
    numerator: dict[str, int]
    denominator: int


class Response(BaseModel):
    success: bool


router = APIRouter()


@router.post('/init-dir', response_model=InitDirResponse)
async def init_dir(request: InitDirRequest):
    models_datasets_service: ModelsDatasetsService = Depends(get_models_datasets_service)
    success = await models_datasets_service.init_dir(request.model, request.dataset)
    return InitDirResponse(success=success)


@router.put('/models-datasets', response_model=Response)
async def models_datasets(request: ModelDatasetI):
    models_datasets_service: ModelsDatasetsService = Depends(get_models_datasets_service)
    success = await models_datasets_service.full_init_dir(
        request.model, request.dataset
    )
    return Response(success=success)


@router.get('/numerator-denominator', response_model=NumeratorDenominator)
async def numerator_denominator(model: str, dataset: str, label: str):
    models_datasets_service: ModelsDatasetsService = Depends(
                                    get_models_datasets_service)
    numerator, denominator = await models_datasets_service.numerator_denominator(
        model, dataset, label
    )
    return NumeratorDenominator(numerator=numerator, denominator=denominator)
