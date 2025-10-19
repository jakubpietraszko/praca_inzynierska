from fastapi import APIRouter, Depends

from interfaces.masks import MasksRequest, MasksResponse
from interfaces.super_pixels import SuperPixelsRequest, SuperPixelsResponse
from services.container import (
    get_masks_service,
)
from services.masks_service import MasksService

router = APIRouter()


@router.post('/rise', response_model=MasksResponse)
async def rise(request: MasksRequest):
    masks_service: MasksService = Depends(get_masks_service)
    return await masks_service.rise_with_save(
        request=request
    )


@router.post('/seed', response_model=SuperPixelsResponse)
async def seed(request: SuperPixelsRequest):
    masks_service: MasksService = Depends(get_masks_service)
    return await masks_service.seed_with_save(
        request=request
    )


@router.post('/slic', response_model=SuperPixelsResponse)
async def slic(request: SuperPixelsRequest):
    masks_service: MasksService = Depends(get_masks_service)
    return await masks_service.slic_with_save(
        request=request
    )
