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


@router.post('/smooth-gradient', response_model=GradientResponse)
async def smooth_grad(request: GradientRequest,
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
        constants.SMOOTH_GRADIENT,
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

        result = torch.zeros_like(image_tensor)

        for _ in range(constants.NUM_SAMPLES):
            noise = torch.randn_like(image_tensor) * constants.NOISE_LEVEL
            noisy_image = (image_tensor + noise).clone().detach().contiguous()
            noisy_image.requires_grad_()
            output = model_info.model(noisy_image)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, noisy_image)[0]
            grad = torch.clamp(grad, min=0)
            result += grad
        result /= constants.NUM_SAMPLES
        result = result.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, constants.AGGREGATION)
        result = utilities_service.relu_normalize(
            result, model_info.input_shape, epsilon=constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, result)
    return GradientResponse(
        success=True,
    )

@router.post('/smooth-gradient-square', response_model=GradientResponse)
async def smooth_grad_square(request: GradientRequest,
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
        constants.SMOOTH_GRADIENT_SQUARE,
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

        result = torch.zeros_like(image_tensor)

        for _ in range(constants.NUM_SAMPLES):
            noise = torch.randn_like(image_tensor) * constants.NOISE_LEVEL
            noisy_image = (image_tensor + noise).clone().detach().contiguous()
            noisy_image.requires_grad_()
            output = model_info.model(noisy_image)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, noisy_image)[0]
            grad = torch.clamp(grad, min=0)
            result += torch.square(grad)
        result /= constants.NUM_SAMPLES
        result = result.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, constants.AGGREGATION)
        result = utilities_service.relu_normalize(
            result, model_info.input_shape, epsilon=constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, result)
    return GradientResponse(
        success=True,
    )

@router.post('/var-gradient', response_model=GradientResponse)
async def var_grad(request: GradientRequest,
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
        constants.VAR_GRADIENT,
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

        results_list = []

        for _ in range(constants.NUM_SAMPLES):
            noise = torch.randn_like(image_tensor) * constants.NOISE_LEVEL
            noisy_image = (image_tensor + noise).clone().detach().contiguous()
            noisy_image.requires_grad_()
            output = model_info.model(noisy_image)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, noisy_image)[0]
            grad = torch.clamp(grad, min=0)
            results_list.append(grad)

        all_grads = torch.stack(results_list, dim=0)
        result = torch.var(all_grads, dim=0)
        result = result.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, constants.AGGREGATION)

        result = utilities_service.relu_normalize(
            result, model_info.input_shape, epsilon=constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, result)
    return GradientResponse(
        success=True,
    )

