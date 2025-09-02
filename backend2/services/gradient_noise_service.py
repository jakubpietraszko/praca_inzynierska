import os
import cv2
import numpy as np
import torch

from interfaces.gradient import GradientRequest, GradientResponse

class GradientNoiseService:
    def __init__(self):
        pass

    def smooth_grad(self,
                 request: GradientRequest,
                 root_path: str,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
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
    
    def smooth_grad_squared(self,
                 request: GradientRequest,
                 root_path: str,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
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
    
    def var_grad(self,
                 request: GradientRequest,
                 root_path: str,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
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