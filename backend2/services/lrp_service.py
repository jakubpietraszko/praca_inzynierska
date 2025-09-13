import os
import numpy as np
from interfaces.gradient import GradientRequest, GradientResponse
from services.models_service import ModelsService
import torch
import cv2


class LRPService:
    def __init__(self):
        pass

    def lrp_flat(self, request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 models_service,
                 constants):

        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_FLAT
        )
        os.makedirs(root_path, exist_ok=True)


        return GradientResponse(
            success=True
        )
