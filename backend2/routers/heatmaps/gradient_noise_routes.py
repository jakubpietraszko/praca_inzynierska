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
from services.container import get_heatmaps_facede_service, HeatmapsFacedeService


router = APIRouter()


@router.post('/smooth-gradient', response_model=GradientResponse)
async def smooth_grad(request: GradientRequest,
                      constants: Constants = Depends(get_constants),
                      heatmaps_facede_service: HeatmapsFacedeService = Depends(
                          get_heatmaps_facede_service
                      ),
                      ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.SMOOTH_GRADIENT,
    )


@router.post('/smooth-gradient-square', response_model=GradientResponse)
async def smooth_grad_square(request: GradientRequest,
                             constants: Constants = Depends(get_constants),
                             heatmaps_facede_service: HeatmapsFacedeService = Depends(
                                 get_heatmaps_facede_service
                             ),
                             ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.SMOOTH_GRADIENT_SQUARE,
    )


@router.post('/var-gradient', response_model=GradientResponse)
async def var_grad(request: GradientRequest,
                   constants: Constants = Depends(get_constants),
                   heatmaps_facede_service: HeatmapsFacedeService = Depends(
                       get_heatmaps_facede_service
                   ),
                   ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.VAR_GRADIENT,
    )
