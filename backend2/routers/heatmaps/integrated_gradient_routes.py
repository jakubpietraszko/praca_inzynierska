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


router = APIRouter()


@router.post('/integrated-gradient', response_model=GradientResponse)
async def integrated_gradient(request: GradientRequest,
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
                                integrated_gradients_service: IntegratedGradientsService = Depends(
                                    get_integrated_gradients_service
                                )
                              ):
    root_path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.INTEGRATED_GRADIENT
    )
    response = integrated_gradients_service.integrated_gradient(
        request=request,
        root_path=root_path,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )

    return response