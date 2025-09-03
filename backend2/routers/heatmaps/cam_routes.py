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
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/grad-cam', response_model=GradientResponse)
async def grad_cam(request: GradientRequest,
                   container: ServiceContainer = Depends(get_container),
                   models_service: ModelsService = Depends(get_models_service),
                   heatmaps_service: HeatmapsService = Depends(
                       get_heatmaps_service),
                   utilities_service: UtilitiesService = Depends(
                       get_utilities_service),
                   models_datasets_service: ModelsDatasetsService = Depends(
                       get_models_datasets_service),
                   datasets_service: DatasetsService = Depends(
                       get_datasets_service),
                   constants: Constants = Depends(get_constants),
                   cam_service: CAMService = Depends(get_cam_service)
                   ):
    response = cam_service.grad_cam(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/grad-cam-plus-plus', response_model=GradientResponse)
async def grad_cam_plus(request: GradientRequest,
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
                        cam_service: CAMService = Depends(get_cam_service)
                        ):
    response = cam_service.grad_cam_plus_plus(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/group-cam', response_model=GradientResponse)
async def group_cam(request: GradientRequest,
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
                    cam_service: CAMService = Depends(get_cam_service)
                    ):
    response = cam_service.group_cam(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/hires-cam', response_model=GradientResponse)
async def hires_cam(request: GradientRequest,
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
                    cam_service: CAMService = Depends(get_cam_service)
                    ):
    response = cam_service.hires_cam(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/wgrad-cam', response_model=GradientResponse)
async def wgrad_cam(request: GradientRequest,
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
                    cam_service: CAMService = Depends(get_cam_service)
                    ):
    response = cam_service.wgrad_cam(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response


@router.post('/ablation-cam', response_model=GradientResponse)
async def ablation_cam(request: GradientRequest,
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
                       cam_service: CAMService = Depends(get_cam_service)
                       ):
    response = cam_service.ablation_cam(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    return response

