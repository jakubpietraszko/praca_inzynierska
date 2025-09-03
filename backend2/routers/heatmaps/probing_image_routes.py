import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService, ProbingImageService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants, get_probing_image_service
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/xrai', response_model=GradientResponse)
async def xrai(request: GradientRequest,
               container: ServiceContainer = Depends(
                   get_container),
               models_service: ModelsService = Depends(
                   get_models_service),
               heatmaps_service: HeatmapsService = Depends(
                   get_heatmaps_service),
               utilities_service: UtilitiesService = Depends(
                   get_utilities_service),
               models_datasets_service: ModelsDatasetsService = Depends(
                   get_models_datasets_service),
               datasets_service: DatasetsService = Depends(
                   get_datasets_service),
               constants: Constants = Depends(get_constants),
               probing_image_service: ProbingImageService = Depends(
                   get_probing_image_service
               )
               ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.XRAI
    )
    response = probing_image_service.xrai(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/rise', response_model=GradientResponse)
async def rise(request: GradientRequest,
               container: ServiceContainer = Depends(
                   get_container),
               models_service: ModelsService = Depends(
                   get_models_service),
               heatmaps_service: HeatmapsService = Depends(
                   get_heatmaps_service),
               utilities_service: UtilitiesService = Depends(
                   get_utilities_service),
               models_datasets_service: ModelsDatasetsService = Depends(
                   get_models_datasets_service),
               datasets_service: DatasetsService = Depends(
                   get_datasets_service),
               constants: Constants = Depends(get_constants),
               probing_image_service: ProbingImageService = Depends(
                   get_probing_image_service
               )
               ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.RISE
    )
    response = probing_image_service.rise(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/exact-shap', response_model=GradientResponse)
async def exact_shap(request: GradientRequest,
                     container: ServiceContainer = Depends(
                         get_container),
                     utilities_service: UtilitiesService = Depends(
                         get_utilities_service),
                     constants: Constants = Depends(get_constants),
                     models_service: ModelsService = Depends(
                         get_models_service),
                     datasets_service: DatasetsService = Depends(
                         get_datasets_service),
                     models_datasets_service: ModelsDatasetsService = Depends(
                         get_models_datasets_service),
                     probing_image_service: ProbingImageService = Depends(
                         get_probing_image_service
                     )
                     ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.EXACT_SHAP
    )
    response = probing_image_service.exact_shap(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/kernel-shap', response_model=GradientResponse)
async def kernel_shap(request: GradientRequest,
                      container: ServiceContainer = Depends(
                          get_container),
                      utilities_service: UtilitiesService = Depends(
                          get_utilities_service),
                      constants: Constants = Depends(get_constants),
                      models_service: ModelsService = Depends(
                          get_models_service),
                      datasets_service: DatasetsService = Depends(
                          get_datasets_service),
                      models_datasets_service: ModelsDatasetsService = Depends(
                          get_models_datasets_service),
                      probing_image_service: ProbingImageService = Depends(
                          get_probing_image_service
                      )
                      ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.KERNEL_SHAP
    )
    response = probing_image_service.kernel_shap(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/lime', response_model=GradientResponse)
async def lime(request: GradientRequest,
               container: ServiceContainer = Depends(
                   get_container),
               utilities_service: UtilitiesService = Depends(
                   get_utilities_service),
               constants: Constants = Depends(get_constants),
               models_service: ModelsService = Depends(
                   get_models_service),
               datasets_service: DatasetsService = Depends(
                   get_datasets_service),
               models_datasets_service: ModelsDatasetsService = Depends(
                   get_models_datasets_service),
               probing_image_service: ProbingImageService = Depends(
                   get_probing_image_service
               )
               ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.LIME
    )
    response = probing_image_service.lime(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response
