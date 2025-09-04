import os
import cv2
import torch
from interfaces.gradient import GradientRequest, GradientResponse


class IntegratedGradientsService:
    def __init__(self):
        pass

    def integrated_gradient(self,
                            request: GradientRequest,
                            container,
                            datasets_service,
                            utilities_service,
                            constants,
                            models_service
                            ) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.INTEGRATED_GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target

            label_path = os.path.join(
                root_path,
                label,
            )
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                utilities_service.add_png_extension(filename)
            )

            if utilities_service.skip_forward(save_path):
                continue
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
                result = utilities_service.aggregate_channels(
                    result, constants.AGGREGATION)

            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=constants.EPSILON
            )

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True
        )
