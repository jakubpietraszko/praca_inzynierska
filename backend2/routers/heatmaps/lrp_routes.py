import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt
from interfaces.gradient import GradientRequest, GradientResponse
from services.heatmaps_facede_service import HeatmapsFacedeService
from services.container import get_heatmaps_facede_service
from services.container import get_constants

router = APIRouter()


@router.post('/lrp-flat', response_model=GradientResponse)
async def gradient(request: GradientRequest,
                   constants: Constants = Depends(get_constants),
                   heatmaps_facede_service: HeatmapsFacedeService = Depends(
                       get_heatmaps_facede_service
                   )
                   ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_FLAT,
    )
