import os
import numpy as np
import asyncio
from interfaces.gradient import GradientRequest, GradientResponse, GradientStatisticRequest, GradientStatisticResponse
from services.models_service import ModelInfo
import torch
import cv2


class GradientService:
    def __init__(self):
        from services.container import get_config_service, get_websocket_service
        from services.container import get_datasets_service
        from services.datasets_service import DatasetsService
        from services.container import get_utilities_service
        from services.utilities_service import UtilitiesService
        from services.models_service import ModelsService
        from services.container import get_models_service
        from services.container import Constants, get_constants
        from services.container import get_container
        from services.container import ServiceContainer

        self.config_service = get_config_service()
        self.websocket_service = get_websocket_service()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.models_service: ModelsService = get_models_service()
        self.constants: Constants = get_constants()
        self.container: ServiceContainer = get_container()

    async def gradient(self, request: GradientRequest) -> GradientResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)

        config = await self.config_service.get_method_config('gradient')
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        print(EPSILON, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path, label
            )

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await self.utilities_service.add_png_extension(filename)
            )

            if await self.utilities_service.skip_forward(save_path):
                # print(f"Skipping {save_path}")
                continue
            image_path = os.path.join(
                self.datasets_service.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await self.datasets_service.load_image(
                image_path, model_info.input_shape
            )

            image_tensor = await self.utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)

            try:
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="gradient"
                )
            except Exception as e:
                pass

        return GradientResponse(
            success=True
        )

    async def gradient_squared(self, request: GradientRequest,
                               container,
                               datasets_service,
                               utilities_service,
                               models_service,
                               constants
                               ):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(request.model, request.dataset),
            constants.GRADIENT_SQUARED
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)

        config = await self.config_service.get_method_config(constants.GRADIENT_SQUARED)
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        print(EPSILON, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path, label
            )

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if await utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await datasets_service.load_image(
                image_path, model_info.input_shape
            )

            image_tensor = await utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = np.square(result)

            result = await utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)
            try:
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="gradient_squared"
                )
            except Exception as e:
                pass

        return GradientResponse(
            success=True
        )

    async def deconvolution(self, request: GradientRequest,
                            container,
                            datasets_service,
                            utilities_service,
                            models_service,
                            constants
                            ):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(request.model, request.dataset),
            constants.DECONVOLUTION
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)

        config = await self.config_service.get_method_config('deconvolution')
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        print(EPSILON, AGGREGATION)

        hooks = list()
        original_inplace_states = dict()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        def deconv(module, grad_input, grad_output):
            if grad_output[0] is not None:
                return (torch.clamp(grad_output[0], min=0.0),)
            return grad_output

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_backward_hook(deconv))

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path, label
            )

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if await utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(
                container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await datasets_service.load_image(
                image_path, model_info.input_shape
            )

            image_tensor = await utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = await utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)
            try:
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="deconvolution"
                )
            except Exception as e:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def guided_backpropagation(self, request: GradientRequest,
                                     container,
                                     datasets_service,
                                     utilities_service,
                                     models_service,
                                     constants
                                     ):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(request.model, request.dataset),
            constants.GUIDED_BACKPROPAGATION
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        config = await self.config_service.get_method_config('guided_backpropagation')
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        print(EPSILON, AGGREGATION)
        forward_activations = dict()
        hooks = list()
        original_inplace_states = dict()
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
                        (grad_input[0] > 0) & (forward_activation > 0),
                        grad_input[0],
                        torch.zeros_like(grad_input[0])
                    )
                    return (guided_grad,)
                return grad_input
            return hook
        for i, (name, module) in enumerate(model_info.model.named_modules()):
            if isinstance(module, torch.nn.ReLU):
                forward_hook_handle = module.register_forward_hook(
                    forward_hook())
                backward_hook_handle = module.register_full_backward_hook(
                    guided_relu_hook())
                hooks.append(forward_hook_handle)
                hooks.append(backward_hook_handle)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path, label
            )

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if await utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await datasets_service.load_image(
                image_path, model_info.input_shape
            )

            image_tensor = await utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = await utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)
            try:
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="guided_backpropagation"
                )
            except Exception as e:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def input_gradient(self, request: GradientRequest,
                             container,
                             datasets_service,
                             utilities_service,
                             models_service,
                             constants
                             ):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(request.model, request.dataset),
            constants.INPUT_GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)

        config = await self.config_service.get_method_config('input_gradient')
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        print(EPSILON, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path, label
            )

            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if await utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await datasets_service.load_image(
                image_path, model_info.input_shape
            )
            image_tensor = await utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()

            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = await utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            input_np = (image_tensor.squeeze().cpu().detach().numpy()
                        * 255).astype(np.uint8)
            if len(input_np.shape) == 3 and input_np.shape[0] == 3:
                input_np = await utilities_service.aggregate_channels(
                    input_np, AGGREGATION)

            result = (result.astype(np.float32) / 255.0) * \
                (input_np.astype(np.float32) / 255.0)
            result = (result * 255.0).astype(np.uint8)

            cv2.imwrite(save_path, result)
            try:
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="input_gradient"
                )
            except Exception as e:
                pass

        return GradientResponse(
            success=True
        )
