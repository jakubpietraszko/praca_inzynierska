import os
import cv2
import numpy as np
import torch

from interfaces.gradient import GradientRequest, GradientResponse

class GradientNoiseService:
    def __init__(self):
        from services.container import get_config_service
        self.config_service = get_config_service()

    def smooth_grad(self,
                 request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.SMOOTH_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        EPSILON = self.config_service.get_method_config('smooth_grad').epsilon
        NUM_SAMPLES = self.config_service.get_method_config('smooth_grad').n_samples
        NOISE = self.config_service.get_method_config('smooth_grad').noise_level
        AGGREGATION = self.config_service.get_method_config('smooth_grad').aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = utilities_service.fill_target(model_info, label)
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
                result = utilities_service.aggregate_channels(
                    result, AGGREGATION)
            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True,
        )
    
    def smooth_grad_squared(self,
                 request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.SMOOTH_GRADIENT_SQUARE,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        EPSILON = self.config_service.get_method_config('smooth_grad_squared').epsilon
        NUM_SAMPLES = self.config_service.get_method_config('smooth_grad_squared').n_samples
        NOISE = self.config_service.get_method_config('smooth_grad_squared').noise_level
        AGGREGATION = self.config_service.get_method_config('smooth_grad_squared').aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = utilities_service.fill_target(model_info, label)
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
                result = utilities_service.aggregate_channels(
                    result, AGGREGATION)
            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True,
        )
    
    def var_grad(self,
                 request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.VAR_GRADIENT,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        EPSILON = self.config_service.get_method_config('var_grad').epsilon
        NUM_SAMPLES = self.config_service.get_method_config('var_grad').n_samples
        NOISE = self.config_service.get_method_config('var_grad').noise_level
        AGGREGATION = self.config_service.get_method_config('var_grad').aggregation

        print(EPSILON, NUM_SAMPLES, NOISE, AGGREGATION)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = utilities_service.fill_target(model_info, label)
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
                result = utilities_service.aggregate_channels(
                    result, AGGREGATION)

            result = utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=EPSILON
            )

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True,
        )