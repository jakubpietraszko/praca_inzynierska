import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService, IntegratedGradientsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants, get_integrated_gradients_service
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt
from services.container import get_heatmaps_facede_service, HeatmapsFacedeService


router = APIRouter()


@router.post('/integrated-gradient', response_model=GradientResponse)
async def integrated_gradient(request: GradientRequest,
                              constants: Constants = Depends(get_constants),
                              heatmaps_facede_service: HeatmapsFacedeService = Depends(
                                  get_heatmaps_facede_service
                              ),
                              ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.INTEGRATED_GRADIENT,
    )
