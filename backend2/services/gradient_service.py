import contextlib
import os

import cv2
import numpy as np
import torch

from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


class GradientService:
    def __init__(self):
        from services.container import (
            Constants,
            LoggerService,
            ServiceContainer,
            UtilitiesService,
            get_config_service,
            get_constants,
            get_container,
            get_datasets_service,
            get_logger_service,
            get_models_service,
            get_utilities_service,
            get_websocket_service,
        )
        from services.datasets_service import DatasetsService
        from services.models_service import ModelsService

        self.config_service = get_config_service()
        self.websocket_service = get_websocket_service()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.models_service: ModelsService = get_models_service()
        self.constants: Constants = get_constants()
        self.container: ServiceContainer = get_container()
        self.logger_service: LoggerService = get_logger_service()
        self.models_datasets_path = self.container.models_datasets_path

    async def gradient(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.GRADIENT)

        self.logger_service.log_config(config, self.constants.GRADIENT)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(root_path, label)

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.datasets_service.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, config.aggregation or self.constants.DEFAULT_AGGREGATION
                )

            result = await self.utilities_service.relu_normalize(result,
                                                                 model_info.input_shape,
                                                                 epsilon=config.epsilon or
                                                                 self.constants.DEFAULT_EPSILON)

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.GRADIENT,
            )


        return HeatmapResponse(success=True)

    async def gradient_squared(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.GRADIENT_SQUARED,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.GRADIENT_SQUARED)

        self.logger_service.log_config(config, self.constants.GRADIENT_SQUARED)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(root_path, label)

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.datasets_service.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(result, config.aggregation or self.constants.DEFAULT_AGGREGATION)

            result = np.square(result)

            result = await self.utilities_service.relu_normalize(result, model_info.input_shape, epsilon=config.epsilon or self.constants.DEFAULT_EPSILON)

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.GRADIENT_SQUARED,
            )

        return HeatmapResponse(success=True)

    async def deconvolution(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.DECONVOLUTION,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.DECONVOLUTION)

        self.logger_service.log_config(config, self.constants.DECONVOLUTION)

        hooks = []
        original_inplace_states = {}

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        def deconv(module, grad_input, grad_output):
            if grad_output[0] is not None:
                return (torch.clamp(grad_output[0], min=0.0),)
            return grad_output

        for _, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_backward_hook(deconv))

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(root_path, label)

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.datasets_service.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(result, config.aggregation or self.constants.DEFAULT_AGGREGATION)

            result = await self.utilities_service.relu_normalize(result, model_info.input_shape, epsilon=config.epsilon or self.constants.DEFAULT_EPSILON)

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.DECONVOLUTION,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return HeatmapResponse(success=True)

    async def guided_backpropagation(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.GUIDED_BACKPROPAGATION,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)
        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.GUIDED_BACKPROPAGATION)

        self.logger_service.log_config(config, self.constants.GUIDED_BACKPROPAGATION)

        hooks = []
        original_inplace_states = {}
        stack = []

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        def forward_hook():
            def hook(module, input, output):
                stack.append(output.clone())

            return hook

        def guided_relu_hook():
            def hook(module, grad_input, grad_output):
                nonlocal stack
                if grad_input[0] is not None and stack[-1] is not None:
                    forward_activation = stack[-1]
                    stack = stack[:-1]
                    guided_grad = torch.where(
                        (grad_input[0] > 0) & (forward_activation > 0), grad_input[0], torch.zeros_like(grad_input[0])
                    )
                    return (guided_grad,)
                return grad_input

            return hook

        for _, (_, module) in enumerate(model_info.model.named_modules()):
            if isinstance(module, torch.nn.ReLU):
                forward_hook_handle = module.register_forward_hook(forward_hook())
                backward_hook_handle = module.register_full_backward_hook(guided_relu_hook())
                hooks.append(forward_hook_handle)
                hooks.append(backward_hook_handle)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(root_path, label)

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(result, config.aggregation or self.constants.DEFAULT_AGGREGATION)

            result = await self.utilities_service.relu_normalize(result, model_info.input_shape, epsilon=config.epsilon or self.constants.DEFAULT_EPSILON)

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.GUIDED_BACKPROPAGATION,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return HeatmapResponse(success=True)

    async def input_gradient(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.INPUT_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.INPUT_GRADIENT)

        self.logger_service.log_config(config, self.constants.INPUT_GRADIENT)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(root_path, label)

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(result, config.aggregation or self.constants.DEFAULT_AGGREGATION)

            result = await self.utilities_service.relu_normalize(result, model_info.input_shape, epsilon=config.epsilon or self.constants.DEFAULT_EPSILON)

            input_np = (image_tensor.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
            if len(input_np.shape) == 3 and input_np.shape[0] == 3:
                input_np = await self.utilities_service.aggregate_channels(input_np, config.aggregation or self.constants.DEFAULT_AGGREGATION)

            result = (result.astype(np.float32) / 255.0) * (input_np.astype(np.float32) / 255.0)
            result = (result * 255.0).astype(np.uint8)

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.INPUT_GRADIENT,
            )

        return HeatmapResponse(success=True)
