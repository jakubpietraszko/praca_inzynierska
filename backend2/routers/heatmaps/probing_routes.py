import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends

from interfaces.gradient import GradientRequest, GradientResponse
from services.container import Constants, get_constants
from services.container import get_heatmaps_facede_service, HeatmapsFacedeService

router = APIRouter()


@router.post('/lime', response_model=GradientResponse)
async def lime(request: GradientRequest,
               constants: Constants = Depends(get_constants),
               heatmaps_facede_service: HeatmapsFacedeService = Depends(
                   get_heatmaps_facede_service
               ),):

    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LIME,
    )


@router.post('/xrai', response_model=GradientResponse)
async def xrai(request: GradientRequest,
               constants: Constants = Depends(get_constants),
               heatmaps_facede_service: HeatmapsFacedeService = Depends(
                   get_heatmaps_facede_service
               ),):

    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.XRAI,
    )


@router.post('/kernel-shap', response_model=GradientResponse)
async def kernel_shap(request: GradientRequest,
                      constants: Constants = Depends(get_constants),
                      heatmaps_facede_service: HeatmapsFacedeService = Depends(
                          get_heatmaps_facede_service
                      ),):

    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.KERNEL_SHAP,
    )
