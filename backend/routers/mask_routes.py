import os
import cv2
from fastapi import APIRouter, Depends
from services.container import get_model_service, get_mask_service, get_dataset_service
from services.model_service import ModelService
from services.mask_service import MaskService
from services.dataset_service import DatasetService
from interfaces.init_dir import InitDirRequest, InitDirResponse
from interfaces.masks import MasksRequest, MasksResponse

router = APIRouter()


@router.post('/init-dir', response_model=InitDirResponse)
async def init_dir(
    request: InitDirRequest,
    mask_service: MaskService = Depends(get_mask_service)
):
    success = mask_service.create_directories(request.model, request.dataset)
    return InitDirResponse(success=success)


@router.post('/mask/rise', response_model=MasksResponse)
async def mask_rise(
    request: MasksRequest,
    model_service: ModelService = Depends(get_model_service),
    mask_service: MaskService = Depends(get_mask_service)
):
    try:
        model_info = model_service.load_model(request.model)
        shape = model_info.shape[1:]  # Remove batch dimension

        for image in request.images:
            label, filename = image.label, image.filename
            
            masks = mask_service.generate_rise_masks(
                shape, 
                scale=5, 
                number_of_masks=request.count
            )
            
            mask_service.save_rise_masks(
                masks, 
                request.model, 
                request.dataset, 
                label, 
                filename
            )

        return MasksResponse(success=True)
    except Exception as e:
        print(f"Error generating RISE masks: {e}")
        return MasksResponse(success=False)


@router.post('/mask/slic', response_model=MasksResponse)
async def mask_slic(
    request: MasksRequest,
    model_service: ModelService = Depends(get_model_service),
    mask_service: MaskService = Depends(get_mask_service),
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        model_info = model_service.load_model(request.model)
        dim, height, width = model_info.shape

        for image in request.images:
            label, filename = image.label, image.filename
            
            image_path = os.path.join(
                dataset_service.datasets_path, 
                request.dataset, 
                label, 
                filename
            )
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            image_np = cv2.imread(image_path)
            if image_np is None:
                print(f"Could not load image: {image_path}")
                continue
            
            mask = mask_service.generate_slic_masks(
                image_np, 
                height, 
                width, 
                number_of_superpixels=request.count
            )
            
            mask_service.save_superpixel_mask(
                mask, 
                request.model, 
                request.dataset, 
                label, 
                filename, 
                'slic'
            )

        return MasksResponse(success=True)
    except Exception as e:
        print(f"Error generating SLIC masks: {e}")
        return MasksResponse(success=False)


@router.post('/mask/seed', response_model=MasksResponse)
async def mask_seed(
    request: MasksRequest,
    model_service: ModelService = Depends(get_model_service),
    mask_service: MaskService = Depends(get_mask_service),
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        model_info = model_service.load_model(request.model)
        dim, height, width = model_info.shape

        for image in request.images:
            label, filename = image.label, image.filename
            
            image_path = os.path.join(
                dataset_service.datasets_path, 
                request.dataset, 
                label, 
                filename
            )
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            image_np = cv2.imread(image_path)
            if image_np is None:
                print(f"Could not load image: {image_path}")
                continue
            
            mask = mask_service.generate_seed_masks(
                image_np, 
                height, 
                width, 
                number_of_superpixels=request.count
            )
            
            mask_service.save_superpixel_mask(
                mask, 
                request.model, 
                request.dataset, 
                label, 
                filename, 
                'seed'
            )

        return MasksResponse(success=True)
    except Exception as e:
        print(f"Error generating SEED masks: {e}")
        return MasksResponse(success=False)
