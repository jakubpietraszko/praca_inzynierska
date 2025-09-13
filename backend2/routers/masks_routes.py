from services.utilities_service import UtilitiesService
from fastapi import APIRouter, Depends
from services.container import get_masks_service
from interfaces.label_filename import LabelFilename
from interfaces.masks import MasksRequest, MasksResponse
from interfaces.super_pixels import SuperPixelsRequest, SuperPixelsResponse
from services.masks_service import MasksService
from services.models_service import ModelsService, ModelInfo
from constans import Constants
from services.models_datasets_service import ModelsDatasetsService
from services.container import get_models_service, get_masks_service, get_constants, get_utilities_service, get_container
from services.container import ServiceContainer
import os
import numpy as np
import cv2
from PIL import Image

router = APIRouter()


@router.post('/rise', response_model=MasksResponse)
async def rise(request: MasksRequest,
               masks_service: MasksService = Depends(get_masks_service),
               ):

    return masks_service.rise_with_save(
        request=request
    )


@router.post('/seed', response_model=SuperPixelsResponse)
async def seed(request: SuperPixelsRequest,
               masks_service: MasksService = Depends(get_masks_service),
               ):

    return masks_service.seed_with_save(
        request=request
    )


@router.post('/slic', response_model=SuperPixelsResponse)
async def slic(request: SuperPixelsRequest,
               container: ServiceContainer = Depends(get_container),
               masks_service: MasksService = Depends(get_masks_service),
               models_service: ModelsService = Depends(get_models_service),
               utilities_service: UtilitiesService = Depends(
                   get_utilities_service),
               constants: Constants = Depends(get_constants)):

    return masks_service.slic_with_save(
        request=request
    )
