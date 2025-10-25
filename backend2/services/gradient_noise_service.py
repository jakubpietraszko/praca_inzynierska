import os

import numpy as np
import torch

from interfaces.gradient import GradientRequest, GradientResponse
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


class GradientNoiseService:
    def __init__(self):
        from services.container import (
            ConfigService,
            Constants,
            DatasetsService,
            LoggerService,
            ModelsService,
            ServiceContainer,
            UtilitiesService,
            WebSocketService,
            get_config_service,
            get_constants,
            get_container,
            get_datasets_service,
            get_logger_service,
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
        self.logger_service: LoggerService = get_logger_service()

    async def smooth_grad(self, request: GradientRequest) -> GradientResponse:
        return await self.__gradient_noise_generic(request, self.constants.SMOOTH_GRADIENT)

    async def smooth_grad_squared(self, request: GradientRequest) -> GradientResponse:
        return await self.__gradient_noise_generic(request, self.constants.SMOOTH_GRADIENT_SQUARE)

    async def var_grad(self, request: GradientRequest) -> GradientResponse:
        return await self.__gradient_noise_generic(request, self.constants.VAR_GRADIENT)

    async def __gradient_noise_generic(self, request: GradientRequest, xai_method_type: str) -> GradientResponse:
        root_path: str = await self.utilities_service.make_root_path(
            request=request, xai_method_type=xai_method_type
        )

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(xai_method_type)

        self.logger_service.log_config(config, xai_method_type)

        for image in request.images:
            label, filename, target = image.label, image.filename, image.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = await self.utilities_service.make_label_path(root_path, label)

            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            model_info.model.eval()

            results_list: list[torch.Tensor] = []

            for _ in range(config.n_samples or self.constants.DEFAULT_N_SAMPLES):
                noise = torch.randn_like(image_tensor) * (config.noise_level or self.constants.DEFAULT_NOISE_LEVEL)
                noisy_image = (image_tensor + noise).clone().detach().contiguous()
                noisy_image.requires_grad_()
                output = model_info.model(noisy_image)
                y_c = output[0, target]
                grad = torch.autograd.grad(y_c, noisy_image)[0]
                grad = torch.clamp(grad, min=0)
                results_list.append(grad)

            result_tensor: torch.Tensor = torch.zeros_like(image_tensor)
            result: np.ndarray | None = None

            if xai_method_type == self.constants.SMOOTH_GRADIENT:
                for item in results_list:
                    result_tensor += item
                result_tensor /= config.n_samples or self.constants.DEFAULT_N_SAMPLES
                result = result_tensor.squeeze().cpu().detach().numpy()
            elif xai_method_type == self.constants.SMOOTH_GRADIENT_SQUARE:
                for item in results_list:
                    result_tensor += item.square()
                result_tensor /= config.n_samples or self.constants.DEFAULT_N_SAMPLES
                result = result_tensor.squeeze().cpu().detach().numpy()
            elif xai_method_type == self.constants.VAR_GRADIENT:
                for item in results_list:
                    result_tensor += item
                smooth_grad: torch.Tensor = torch.stack(results_list, dim=0)
                result = torch.var(smooth_grad, dim=0).squeeze().cpu().detach().numpy()

            if result is None:
                raise TypeError

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, config.aggregation or self.constants.DEFAULT_AGGREGATION
                )

            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=config.epsilon or self.constants.DEFAULT_EPSILON
            )

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=xai_method_type,
            )


        return GradientResponse(
            success=True,
        )
