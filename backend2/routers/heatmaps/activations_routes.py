import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService, ActivationsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants, get_activations_service
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()

@router.post('/activations', response_model=ActivationsResponse)
async def activations(request: ActivationsRequest,
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
                      activations_service: ActivationsService = Depends(
                          get_activations_service
                      )
                      ):
    response = activations_service.activations(
        request=request,
        container=container,
        models_service=models_service,
        utilities_service=utilities_service,
        datasets_service=datasets_service,
        constants=constants,
    )

    return response
