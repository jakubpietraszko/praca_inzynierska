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
from services.heatmaps_facede_service import HeatmapsFacedeService
from services.container import get_heatmaps_facede_service

from constans import Constants

from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/gradient', response_model=GradientResponse)
async def gradient(request: GradientRequest,
                   container: ServiceContainer = Depends(get_container),
                   models_service: ModelsService = Depends(get_models_service),
                   utilities_service: UtilitiesService = Depends(
                       get_utilities_service),
                   datasets_service: DatasetsService = Depends(
                       get_datasets_service),
                   constants: Constants = Depends(get_constants),
                   heatmaps_facede_service: HeatmapsFacedeService = Depends(
                       get_heatmaps_facede_service
                   )
                   ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service
    )


@router.post('/gradient-squared', response_model=GradientResponse)
async def gradient_squared(request: GradientRequest,
                           container: ServiceContainer = Depends(
                               get_container),
                           models_service: ModelsService = Depends(
                               get_models_service),
                           utilities_service: UtilitiesService = Depends(
                               get_utilities_service),
                           datasets_service: DatasetsService = Depends(
                               get_datasets_service),
                           constants: Constants = Depends(get_constants),
                           heatmaps_facede_service: HeatmapsFacedeService = Depends(
                               get_heatmaps_facede_service
                           )

                           ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT_SQUARED,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service
    )


@router.post('/deconvolution', response_model=GradientResponse)
async def deconvolution(request: GradientRequest,
                        container: ServiceContainer = Depends(get_container),
                        models_service: ModelsService = Depends(
                            get_models_service),
                        utilities_service: UtilitiesService = Depends(
                            get_utilities_service),
                        datasets_service: DatasetsService = Depends(
                            get_datasets_service),
                        constants: Constants = Depends(get_constants),
                        heatmaps_facede_service: HeatmapsFacedeService = Depends(
                            get_heatmaps_facede_service
                        )
                        ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.DECONVOLUTION,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )


@router.post('/guided-backpropagation', response_model=GradientResponse)
async def guided_backpropagation(request: GradientRequest,
                                 container: ServiceContainer = Depends(
                                     get_container),
                                 models_service: ModelsService = Depends(
                                     get_models_service),
                                 utilities_service: UtilitiesService = Depends(
                                     get_utilities_service),
                                 datasets_service: DatasetsService = Depends(
                                     get_datasets_service),
                                 constants: Constants = Depends(get_constants),
                                 heatmaps_facede_service: HeatmapsFacedeService = Depends(
                                     get_heatmaps_facede_service
                                 )
                                 ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GUIDED_BACKPROPAGATION,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )


@router.post('/input-gradient', response_model=GradientResponse)
async def input_gradient(request: GradientRequest,
                         container: ServiceContainer = Depends(get_container),
                         models_service: ModelsService = Depends(
                             get_models_service),
                         utilities_service: UtilitiesService = Depends(
                             get_utilities_service),
                         datasets_service: DatasetsService = Depends(
                             get_datasets_service),
                         constants: Constants = Depends(get_constants),
                         heatmaps_facede_service: HeatmapsFacedeService = Depends(
                             get_heatmaps_facede_service
                         )
                         ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.INPUT_GRADIENT,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        models_service=models_service,
        constants=constants
    )
