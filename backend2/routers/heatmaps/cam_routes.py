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
                   constants: Constants = Depends(get_constants)
                   ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.GRAD_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=True)
        

        model_info.model.eval()


        activations = None
        gradients = None

        def forward_hook(module, i, o):
            nonlocal activations
            activations = o.clone().detach()

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0].clone().detach()

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)
        model_info.model.eval()
        image_tensor.requires_grad_(True)
        output = model_info.model(image_tensor)
        output[0, target].backward()
        h1.remove()
        h2.remove()



        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = utilities_service.relu_normalize(
            cam, model_info.input_shape, constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, cam)

    return GradientResponse(
        success=True,
    )


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
                        constants: Constants = Depends(get_constants)
                        ):
    pass


@router.post('/group-cam', response_model=GradientResponse)
async def group_cam(request: GradientRequest):
    pass


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
                    constants: Constants = Depends(get_constants)
                    ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.HIRES_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=True)

        model_info.model.eval()

        activations = None
        gradients = None

        def forward_hook(module, i, o):
            nonlocal activations
            activations = o.clone().detach()

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0].clone().detach()

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)

        output = model_info.model(image_tensor)
        output[0, target].backward()
        h1.remove()
        h2.remove()

        weights = gradients

        cam = (activations * weights).sum(dim=1, keepdim=True)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = utilities_service.relu_normalize(
            cam, model_info.input_shape, constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename) 
        )
        cv2.imwrite(save_path, cam)
    return GradientResponse(
        success=True,
    )


@router.post('/wgrad-cam', response_model=GradientResponse)
async def wgrad_cam(request: GradientRequest):
    pass


@router.post('/ablation-cam', response_model=GradientResponse)
async def ablation_cam(request: GradientRequest):
    pass
