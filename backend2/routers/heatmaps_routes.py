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


@router.post('/gradient', response_model=GradientResponse)
async def gradient(request: GradientRequest,
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
        constants.GRADIENT,
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

        output = model_info.model(image_tensor)
        y_c = output[0, target]
        grad = torch.autograd.grad(y_c, image_tensor)[0]
        result = grad.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, 'sum')

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


@router.post('/integrated-gradient', response_model=GradientResponse)
async def integrated_gradient(request: GradientRequest,
                              container: ServiceContainer = Depends(
                                  get_container),
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
        constants.INTEGRATED_GRADIENT,
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

        baseline = torch.ones_like(image_tensor) * constants.BASELINE
        alphas = torch.linspace(0, 1, constants.STEPS).reshape(-1, 1, 1, 1)

        interpolated_images = baseline + alphas * (image_tensor - baseline)

        result = torch.zeros_like(image_tensor)

        for i in range(constants.STEPS):
            interpolated_image = interpolated_images[i].requires_grad_()
            output = model_info.model(interpolated_image.unsqueeze(0))
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, interpolated_image)[0]
            grad = torch.clamp(grad, min=0)
            result += grad
        result /= constants.STEPS
        result = result.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, 'sum')

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
            noisy_image = (image_tensor + noise).clone().detach()
            noisy_image.requires_grad_()
            output = model_info.model(noisy_image)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, noisy_image)[0]
            grad = torch.clamp(grad, min=0)
            result += grad
        result /= constants.NUM_SAMPLES
        result = result.squeeze().cpu().detach().numpy()

        if len(result.shape) == 3 and result.shape[0] == 3:
            result = utilities_service.aggregate_channels(result, 'sum')
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
            result = utilities_service.aggregate_channels(result, 'sum')
        
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
            activations = o

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0]

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)
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
            activations = o

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0]

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_backward_hook(backward_hook)

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


@router.post('/xrai', response_model=GradientResponse)
async def xrai(request: GradientRequest):
    pass


@router.post('/rise', response_model=GradientResponse)
async def rise(request: GradientRequest):
    pass


@router.post('/exact-shap', response_model=GradientResponse)
async def exact_shap(request: GradientRequest):
    pass


@router.post('/kernel-shap', response_model=GradientResponse)
async def kernel_shap(request: GradientRequest):
    pass


@router.post('/lime', response_model=GradientResponse)
async def lime(request: GradientRequest):
    pass
