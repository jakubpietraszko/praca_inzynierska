import os
import cv2
import torch
from interfaces.gradient import GradientRequest, GradientResponse


class IntegratedGradientsService:
    def __init__(self):
        from services.container import get_config_service, get_websocket_service
        self.config_service = get_config_service()
        self.websocket_service = get_websocket_service()

    async def integrated_gradient(self,
                            request: GradientRequest,
                            container,
                            datasets_service,
                            utilities_service,
                            constants,
                            models_service
                            ) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(request.model, request.dataset),
            constants.INTEGRATED_GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)

        config = await self.config_service.get_method_config('integrated_gradient')
        EPSILON = config.epsilon
        AGGREGATION = config.aggregation
        BASELINE = config.baseline
        STEPS = config.steps
        print(EPSILON, AGGREGATION, BASELINE, STEPS)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await utilities_service.fill_target(model_info, label)
            label_path = os.path.join(
                root_path,
                label,
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
                image_path, model_info.input_shape)
            image_tensor = await utilities_service.convert_image_to_tensor(
                image_np, grad=True)
            model_info.model.eval()

            baseline = torch.ones_like(image_tensor) * BASELINE
            alphas = torch.linspace(0, 1, STEPS).reshape(-1, 1, 1, 1)

            interpolated_images = baseline + alphas * (image_tensor - baseline)

            result = torch.zeros_like(image_tensor)

            for i in range(STEPS):
                interpolated_image = interpolated_images[i].requires_grad_()
                output = model_info.model(interpolated_image.unsqueeze(0))
                y_c = output[0, target]
                grad = torch.autograd.grad(y_c, interpolated_image)[0]
                grad = torch.clamp(grad, min=0)
                result += grad
            result /= STEPS
            result = result.squeeze().cpu().detach().numpy()

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
                    heatmap_type="integrated_gradient",
                )
            except:
                pass

        return GradientResponse(
            success=True
        )
