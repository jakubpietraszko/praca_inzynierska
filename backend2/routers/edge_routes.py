import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.edge import EdgeRequest, EdgeResponse
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()

@router.post('/edge-canny')
async def canny_edge(edge_request: EdgeRequest,
                        container: ServiceContainer = Depends(get_container),
                        models_service: ModelsService = Depends(get_models_service),
                        heatmaps_service: HeatmapsService = Depends(get_heatmaps_service),
                        utilities_service: UtilitiesService = Depends(get_utilities_service),
                        models_datasets_service: ModelsDatasetsService = Depends(get_models_datasets_service),
                        datasets_service: DatasetsService = Depends(get_datasets_service),
                        constants: Constants = Depends(get_constants)
                     ):
    model_info = models_service.load_model(edge_request.model)

    for label_filename in edge_request.images:
        label = label_filename.label
        filename = label_filename.filename

        models_datasets_name = utilities_service.models_datasets_path(
            edge_request.model, edge_request.dataset
        )

        path = os.path.join(
            container.models_datasets_path,
            models_datasets_name,
            constants.EDGE_CANNY, label
        )

        os.makedirs(path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            edge_request.dataset, label, filename
        )

        image_np = np.array(Image.open(image_path))

        image_np = cv2.resize(
            image_np,
            (model_info.input_shape[2], model_info.input_shape[1])
        )

        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(image_np, 100, 200)

        canny_path = os.path.join(path, filename)

        cv2.imwrite(canny_path, canny)

    return EdgeResponse(success=True)