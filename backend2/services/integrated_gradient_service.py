import builtins
import contextlib
import os

import cv2
import torch

from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


class IntegratedGradientsService:
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
        self.logger_service: LoggerService = get_logger_service()
        self.container: ServiceContainer = get_container()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.models_service: ModelsService = get_models_service()

    async def integrated_gradient(self, request: HeatmapRequest) -> HeatmapResponse:
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.INTEGRATED_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.INTEGRATED_GRADIENT)

        self.logger_service.log_config(config, self.constants.INTEGRATED_GRADIENT)

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
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue
            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)
            model_info.model.eval()

            baseline = torch.ones_like(image_tensor) * (config.baseline or self.constants.DEFAULT_BASELINE)
            alphas = torch.linspace(0, 1, config.steps or self.constants.DEFAULT_STEPS).reshape(-1, 1, 1, 1)

            interpolated_images = baseline + alphas * (image_tensor - baseline)

            result = torch.zeros_like(image_tensor)

            for i in range(config.steps or self.constants.DEFAULT_STEPS):
                interpolated_image = interpolated_images[i].requires_grad_()
                output = model_info.model(interpolated_image.unsqueeze(0))
                y_c = output[0, target]
                grad = torch.autograd.grad(y_c, interpolated_image)[0]
                grad = torch.clamp(grad, min=0)
                result += grad
            result /= config.steps or self.constants.DEFAULT_STEPS
            result = result.squeeze().cpu().detach().numpy()

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
                heatmap_type=self.constants.INTEGRATED_GRADIENT,
            )

        return HeatmapResponse(success=True)
