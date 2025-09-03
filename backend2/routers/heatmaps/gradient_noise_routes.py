import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService, GradientNoiseService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants, get_gradient_noise_service
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/smooth-gradient', response_model=GradientResponse)
async def smooth_grad(request: GradientRequest,
                      container: ServiceContainer = Depends(get_container),
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
                      gradient_noise_service: GradientNoiseService = Depends(
                          get_gradient_noise_service)
                      ):
    response = gradient_noise_service.smooth_grad(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )

    return response

@router.post('/smooth-gradient-square', response_model=GradientResponse)
async def smooth_grad_square(request: GradientRequest,
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
                             gradient_noise_service: GradientNoiseService = Depends(
        get_gradient_noise_service)
):
    response = gradient_noise_service.smooth_grad_squared(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/var-gradient', response_model=GradientResponse)
async def var_grad(request: GradientRequest,
                   container: ServiceContainer = Depends(get_container),
                   models_service: ModelsService = Depends(get_models_service),
                   heatmaps_service: HeatmapsService = Depends(
                       get_heatmaps_service),
                   utilities_service: UtilitiesService = Depends(
                       get_utilities_service),
                   models_datasets_service: ModelsDatasetsService = Depends(
                       get_models_datasets_service),
                   datasets_service: DatasetsService = Depends(
                       get_datasets_service),
                   constants: Constants = Depends(get_constants),
                   gradient_noise_service: GradientNoiseService = Depends(
                       get_gradient_noise_service)
                   ):
    response = gradient_noise_service.var_grad(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response    
