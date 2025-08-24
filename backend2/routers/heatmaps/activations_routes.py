import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
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
                      constants: Constants = Depends(get_constants)
                      ):
    model_info = models_service.load_model(request.model)

    filepath = os.path.join(
        container.datasets_path,
        request.dataset,
        request.label,
        request.filename
    )

    image_np = datasets_service.load_image(filepath, model_info.input_shape)
    image_tensor = utilities_service.convert_image_to_tensor(image_np)

    activations = None

    def hook(module, input, output):
        nonlocal activations
        activations = output.detach().cpu().numpy()

    ref = model_info.last_conv_layer.register_forward_hook(hook)
    model_info.model.eval()
    model_info.model(image_tensor)
    ref.remove()

    activations = np.squeeze(activations, axis=0)

    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ACTIVATIONS,
        request.label,
        utilities_service.delete_extension(request.filename)
    )
    os.makedirs(path, exist_ok=True)

    for i, activation in enumerate(activations):
        activation = utilities_service.relu_normalize(
            activation, model_info.input_shape, epsilon=constants.EPSILON)

        activation_path = os.path.join(
            path,
            f'{i}.png'
        )
        cv2.imwrite(activation_path, activation)

    return ActivationsResponse(
        success=True,
        count=len(activations)
    )