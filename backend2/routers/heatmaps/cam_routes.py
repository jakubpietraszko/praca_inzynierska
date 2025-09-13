import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from sklearn.decomposition import PCA
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService, CAMService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants, get_cam_service
from services.container import get_heatmaps_facede_service, HeatmapsFacedeService
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/grad-cam', response_model=GradientResponse)
async def grad_cam(request: GradientRequest,
                   constants: Constants = Depends(get_constants),
                   heatmaps_facede_service: HeatmapsFacedeService = Depends(
                       get_heatmaps_facede_service
                   )
                   ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRAD_CAM,
    )


@router.post('/grad-cam-plus-plus', response_model=GradientResponse)
async def grad_cam_plus(request: GradientRequest,
                        constants: Constants = Depends(get_constants),
                        heatmaps_facede_service: HeatmapsFacedeService = Depends(
                            get_heatmaps_facede_service
                        )
                        ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRAD_CAM_PLUS_PLUS,
    )


@router.post('/group-cam', response_model=GradientResponse)
async def group_cam(request: GradientRequest,
                    constants: Constants = Depends(get_constants),
                    heatmaps_facede_service: HeatmapsFacedeService = Depends(
                        get_heatmaps_facede_service
                    )
                    ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GROUP_CAM,
    )


@router.post('/hires-cam', response_model=GradientResponse)
async def hires_cam(request: GradientRequest,
                    constants: Constants = Depends(get_constants),
                    heatmaps_facede_service: HeatmapsFacedeService = Depends(
                        get_heatmaps_facede_service
                    )
                    ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.HIRES_CAM,
    )


@router.post('/wgrad-cam', response_model=GradientResponse)
async def wgrad_cam(request: GradientRequest,
                    constants: Constants = Depends(get_constants),
                    heatmaps_facede_service: HeatmapsFacedeService = Depends(
                        get_heatmaps_facede_service
                    )
                    ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.WGRAD_CAM,
    )


@router.post('/ablation-cam', response_model=GradientResponse)
async def ablation_cam(request: GradientRequest,
                       constants: Constants = Depends(get_constants),
                       heatmaps_facede_service: HeatmapsFacedeService = Depends(
                           get_heatmaps_facede_service
                       )
                       ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.ABLATION_CAM,
    )

@router.post('/xgrad-cam', response_model=GradientResponse)
async def xgrad_cam(request: GradientRequest,
                    constants: Constants = Depends(get_constants),
                    heatmaps_facede_service: HeatmapsFacedeService = Depends(
                        get_heatmaps_facede_service
                    )
                    ):
    return heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.XGRAD_CAM,
    )