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
            result = utilities_service.aggregate_channels(
                result, constants.AGGREGATION)

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

@router.post('/gradient-squared', response_model=GradientResponse)
async def gradient_squared(request: GradientRequest,
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
        constants.GRADIENT_SQUARED,
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
            result = utilities_service.aggregate_channels(
                result, constants.AGGREGATION)
            
        result = np.square(result)

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

@router.post('/deconvolution', response_model=GradientResponse)
async def deconvolution(request: GradientRequest,
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
        constants.DECONVOLUTION,
    )
    os.makedirs(path, exist_ok=True)

    hooks = []
    original_inplace_states = {}

    for name, module in model_info.model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            original_inplace_states[name] = module.inplace
            module.inplace = False

    def deconv_relu_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            return (torch.clamp(grad_output[0], min=0),)
        return grad_output

    for name, module in model_info.model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hook = module.register_full_backward_hook(deconv_relu_hook)
            hooks.append(hook)

    try:
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
                result = utilities_service.aggregate_channels(
                    result, constants.AGGREGATION)

            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=constants.EPSILON
            )

            save_path = os.path.join(
                label_path,
                utilities_service.add_png_extension(filename)
            )

            cv2.imwrite(save_path, result)

    finally:
        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU) and name in original_inplace_states:
                module.inplace = original_inplace_states[name]

    return GradientResponse(
        success=True,
    )

@router.post('/guided-backpropagation', response_model=GradientResponse)
async def guided_backpropagation(request: GradientRequest,
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
        constants.GUIDED_BACKPROPAGATION,
    )
    os.makedirs(path, exist_ok=True)

    forward_activations = {}
    hooks = []
    original_inplace_states = {}

    for name, module in model_info.model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            original_inplace_states[name] = module.inplace
            module.inplace = False

    def forward_hook(name):
        def hook(module, input, output):
            forward_activations[name] = input[0].clone() if isinstance(
                input, tuple) else input.clone()
        return hook

    def guided_relu_hook(name):
        def hook(module, grad_input, grad_output):
            if name in forward_activations and grad_output[0] is not None:
                forward_input = forward_activations[name]
                grad_output_tensor = grad_output[0]

                if forward_input.shape != grad_output_tensor.shape:
                    if forward_input.dim() >= 2 and grad_output_tensor.dim() >= 2:
                        target_spatial_size = grad_output_tensor.shape[-2:]
                        forward_spatial_size = forward_input.shape[-2:]

                        if target_spatial_size != forward_spatial_size and len(target_spatial_size) == 2:
                            if forward_input.dim() == 2:
                                forward_input = forward_input.unsqueeze(
                                    0).unsqueeze(0)
                                need_squeeze = True
                            elif forward_input.dim() == 3:
                                forward_input = forward_input.unsqueeze(0)
                                need_squeeze = True
                            else:
                                need_squeeze = False

                            forward_input = torch.nn.functional.interpolate(
                                forward_input,
                                size=target_spatial_size,
                                mode='nearest'
                            )

                            if need_squeeze:
                                if grad_output_tensor.dim() == 2:
                                    forward_input = forward_input.squeeze(
                                        0).squeeze(0)
                                elif grad_output_tensor.dim() == 3:
                                    forward_input = forward_input.squeeze(0)
                    else:
                        return grad_output

                forward_mask = (forward_input > 0).float()
                backward_mask = (grad_output_tensor > 0).float()
                combined_mask = forward_mask * backward_mask
                return (grad_output_tensor * combined_mask,)
            return grad_output
        return hook

    relu_count = 0
    for name, module in model_info.model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            relu_name = f'relu_{relu_count}'
            forward_hook_handle = module.register_forward_hook(
                forward_hook(relu_name))
            backward_hook_handle = module.register_full_backward_hook(
                guided_relu_hook(relu_name))
            hooks.extend([forward_hook_handle, backward_hook_handle])
            relu_count += 1

    try:
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
                result = utilities_service.aggregate_channels(
                    result, constants.AGGREGATION)

            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=constants.EPSILON
            )

            save_path = os.path.join(
                label_path,
                utilities_service.add_png_extension(filename)
            )

            cv2.imwrite(save_path, result)

    finally:
        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU) and name in original_inplace_states:
                module.inplace = original_inplace_states[name]

    return GradientResponse(
        success=True,
    )

@router.post('/input-gradient', response_model=GradientResponse)
async def input_gradient(request: GradientRequest,
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
        constants.INPUT_GRADIENT,
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
            result = utilities_service.aggregate_channels(
                result, constants.AGGREGATION)

        result = utilities_service.relu_normalize(
            result, model_info.input_shape, epsilon=constants.EPSILON
        )

        input_np = (image_tensor.squeeze().cpu().detach().numpy()
                    * 255).astype(np.uint8)
        if len(input_np.shape) == 3 and input_np.shape[0] == 3:
            input_np = utilities_service.aggregate_channels(
                input_np, constants.AGGREGATION)

        result = (result.astype(np.float32) / 255.0) * \
            (input_np.astype(np.float32) / 255.0)
        result = (result * 255.0).astype(np.uint8)

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )

        cv2.imwrite(save_path, result)

    return GradientResponse(
        success=True,
    )
