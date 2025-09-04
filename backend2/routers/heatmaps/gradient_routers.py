import os
import numpy as np
import cv2
import torch

from fastapi import APIRouter, Depends

from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse

from services.container import ServiceContainer, get_container
from services.container import get_models_service, get_heatmaps_service, get_utilities_service, get_heatmaps_service, get_models_datasets_service, get_datasets_service, get_gradient_service
from services.models_service import ModelsService
from services.heatmaps_service import HeatmapsService
from services.utilities_service import UtilitiesService
from services.heatmaps_service import HeatmapsService
from services.models_datasets_service import ModelsDatasetsService
from services.datasets_service import DatasetsService
from services.gradient_service import GradientService
from services.container import get_constants

from constans import Constants

from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/gradient', response_model=GradientResponse)
async def gradient(request: GradientRequest,
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
                   gradients_service: GradientService = Depends(
                       get_gradient_service
                   )
                   ):
    response = gradients_service.gradient(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )

    return response


@router.post('/gradient-squared', response_model=GradientResponse)
async def gradient_squared(request: GradientRequest,
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
                           gradients_service: GradientService = Depends(
                               get_gradient_service
                           )
                           ):
    response = gradients_service.gradient_squared(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )

    return response


@router.post('/deconvolution', response_model=GradientResponse)
async def deconvolution(request: GradientRequest,
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
                        gradients_service: GradientService = Depends(
                            get_gradient_service
                        )
                        ):
    response = gradients_service.deconvolution(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )

    return response


@router.post('/guided-backpropagation', response_model=GradientResponse)
async def guided_backpropagation(request: GradientRequest,
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
                                 gradients_service: GradientService = Depends(
                                     get_gradient_service
                                 )
                                 ):
    response = gradients_service.guided_backpropagation(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )

    return response


@router.post('/input-gradient', response_model=GradientResponse)
async def input_gradient(request: GradientRequest,
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
                         gradients_service: GradientService = Depends(
                             get_gradient_service
                         )
                         ):
    response = gradients_service.input_gradient(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )

    return response
