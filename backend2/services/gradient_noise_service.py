import builtins
import contextlib
import os

import cv2
import torch

from interfaces.gradient import GradientRequest, GradientResponse


class GradientNoiseService:
    def __init__(self):
        from services.container import (
            ConfigService,
            Constants,
            DatasetsService,
            ModelsService,
            ServiceContainer,
            UtilitiesService,
            WebSocketService,
            get_config_service,
            get_constants,
            get_container,
            get_datasets_service,
            get_models_service,
            get_utilities_service,
            get_websocket_service,
        )
        self.config_service: ConfigService = get_config_service()
        self.websocket_service: WebSocketService = get_websocket_service()
        self.constants: Constants = get_constants()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.models_service: ModelsService = get_models_service()
        self.container: ServiceContainer = get_container()

    async def smooth_grad(self, request: GradientRequest) -> GradientResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.SMOOTH_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)

        config = await self.config_service.get_method_config('smooth_grad')
        EPSILON = config.epsilon
        NUM_SAMPLES = config.n_samples
        NOISE = config.noise_level
        AGGREGATION = config.aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path,
                label,
            )
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await self.utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                self.container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await self.datasets_service.load_image(
                image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(
                image_np, grad=True)

            model_info.model.eval()

            result = torch.zeros_like(image_tensor)

            for _ in range(NUM_SAMPLES):
                noise = torch.randn_like(image_tensor) * NOISE
                noisy_image = (image_tensor + noise).clone().detach().contiguous()
                noisy_image.requires_grad_()
                output = model_info.model(noisy_image)
                y_c = output[0, target]
                grad = torch.autograd.grad(y_c, noisy_image)[0]
                grad = torch.clamp(grad, min=0)
                result += grad
            result /= NUM_SAMPLES
            result = result.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, AGGREGATION)
            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)

            with contextlib.suppress(builtins.BaseException):
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="smooth_gradient",
                )

        return GradientResponse(
            success=True,
        )

    async def smooth_grad_squared(self, request: GradientRequest) -> GradientResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.SMOOTH_GRADIENT_SQUARE,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)

        config = await self.config_service.get_method_config('smooth_grad_squared')
        EPSILON = config.epsilon
        NUM_SAMPLES = config.n_samples
        NOISE = config.noise_level
        AGGREGATION = config.aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path,
                label,
            )
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await self.utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                self.container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await self.datasets_service.load_image(
                image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(
                image_np, grad=True)

            model_info.model.eval()

            result = torch.zeros_like(image_tensor)

            for _ in range(NUM_SAMPLES):
                noise = torch.randn_like(image_tensor) * NOISE
                noisy_image = (image_tensor + noise).clone().detach().contiguous()
                noisy_image.requires_grad_()
                output = model_info.model(noisy_image)
                y_c = output[0, target]
                grad = torch.autograd.grad(y_c, noisy_image)[0]
                grad = torch.clamp(grad, min=0)
                result += torch.square(grad)
            result /= NUM_SAMPLES
            result = result.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, AGGREGATION)
            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)
            with contextlib.suppress(builtins.BaseException):
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="smooth_gradient_square",
                )

        return GradientResponse(
            success=True,
        )

    async def var_grad(self, request: GradientRequest) -> GradientResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.VAR_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)

        config = await self.config_service.get_method_config('var_grad')
        EPSILON = config.epsilon
        NUM_SAMPLES = config.n_samples
        NOISE = config.noise_level
        AGGREGATION = config.aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path,
                label,
            )
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await self.utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(
                self.container.datasets_path,
                request.dataset,
                label,
                filename
            )

            image_np = await self.datasets_service.load_image(
                image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(
                image_np, grad=True)

            model_info.model.eval()

            results_list = []

            for _ in range(NUM_SAMPLES):
                noise = torch.randn_like(image_tensor) * NOISE
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
                result = await self.utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)
            with contextlib.suppress(builtins.BaseException):
                await self.websocket_service.send_heatmap_notification_sync(
                    dataset=request.dataset,
                    model=request.model,
                    label=label,
                    filename=filename,
                    heatmap_type="var_gradient",
                )
        return GradientResponse(
            success=True,
        )
