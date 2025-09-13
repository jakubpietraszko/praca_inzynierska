import os
import cv2
import numpy as np
import torch

from interfaces.gradient import GradientRequest, GradientResponse


class ProbingImageService:
    def __init__(self):
        from services.container import MasksService, get_masks_service

        self.get_masks_service = get_masks_service

    def xrai(self, request: GradientRequest,
             container,
             datasets_service,
             utilities_service,
             constants,
             models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )

    def rise(self, request: GradientRequest,
             container,
             datasets_service,
             utilities_service,
             constants,
             models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )

    def exact_shap(self, request: GradientRequest,
                   container,
                   datasets_service,
                   utilities_service,
                   constants,
                   models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )

    def exact_shap(self, request: GradientRequest,
                   container,
                   datasets_service,
                   utilities_service,
                   constants,
                   models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )

    def kernel_shap(self, request: GradientRequest,
                    container,
                    datasets_service,
                    utilities_service,
                    constants,
                    models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )

    def lime(self, request: GradientRequest,
             container,
             datasets_service,
             utilities_service,
             constants,
             models_service) -> GradientResponse:

        return GradientResponse(
            success=True
        )
