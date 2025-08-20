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
               container: ServiceContainer = Depends(get_container),
               masks_service: MasksService = Depends(get_masks_service),
               models_service: ModelsService = Depends(get_models_service),
               utilities_service: UtilitiesService = Depends(get_utilities_service),
               ):
    model_info: ModelInfo = models_service.load_model(request.model)


    for label_filename in request.images:
        label = label_filename.label
        filename = label_filename.filename

        models_datasets_name = utilities_service.models_datasets_path(
            request.model, request.dataset
        )

        path = os.path.join(
            container.models_datasets_path,
            models_datasets_name, 
            'mask_rise', label, filename.split('.')[0]
        )


        os.makedirs(path, exist_ok=True)

        masks = masks_service.rise(
            height=model_info.input_shape[1],
            width=model_info.input_shape[2],
            scale=5,
            number_of_random_mask=request.count
        )

        for i, mask in enumerate(masks):
            mask_path = os.path.join(path, f'{i}.png')
            cv2.imwrite(mask_path, mask * 255)

    return MasksResponse(success=True)

@router.post('/seed', response_model=SuperPixelsResponse)
async def seed(request: SuperPixelsRequest,
               container: ServiceContainer = Depends(get_container),
               masks_service: MasksService = Depends(get_masks_service),
               models_service: ModelsService = Depends(get_models_service),
               utilities_service: UtilitiesService = Depends(get_utilities_service),
               constants: Constants = Depends(get_constants)):
    model_info: ModelInfo = models_service.load_model(request.model)
    
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        'mask_seed',
    )
    os.makedirs(path, exist_ok=True)
    counts = list()

    for label_filename in request.images:
        label = label_filename.label
        filename = label_filename.filename

        label_path = os.path.join(path, label)
        os.makedirs(label_path, exist_ok=True)

        image_path_save = os.path.join(label_path, filename)
        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )
        
        image_np = np.array(Image.open(image_path))
        dim = model_info.input_shape[0]
        height = model_info.input_shape[1]
        width = model_info.input_shape[2]
        mask, count = masks_service.seed(image_np,
                                  height,
                                  width,
                                  dim,
                                  super_pixels=request.superPixels)
        counts.append(count)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
        cv2.imwrite(image_path_save, mask)

    return SuperPixelsResponse(success=True, superPixels=counts)

        

@router.post('/slic', response_model=SuperPixelsResponse)
async def slic(request: SuperPixelsRequest,
               container: ServiceContainer = Depends(get_container),
               masks_service: MasksService = Depends(get_masks_service),
               models_service: ModelsService = Depends(get_models_service),
               utilities_service: UtilitiesService = Depends(get_utilities_service),
               constants: Constants = Depends(get_constants)):
    model_info: ModelInfo = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        'mask_slic',
    )
    os.makedirs(path, exist_ok=True)
    counts = list()

    for label_filename in request.images:
        label = label_filename.label
        filename = label_filename.filename

        label_path = os.path.join(path, label)
        os.makedirs(label_path, exist_ok=True)

        image_path_save = os.path.join(label_path, filename)
        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )
        
        image_np = np.array(Image.open(image_path))
        dim = model_info.input_shape[0]
        height = model_info.input_shape[1]
        width = model_info.input_shape[2]
        mask, count = masks_service.slic(image_np,
                                  height,
                                  width,
                                  dim,
                                  super_pixels=request.superPixels)
        counts.append(count)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
        cv2.imwrite(image_path_save, mask)

    return SuperPixelsResponse(success=True, superPixels=counts)
