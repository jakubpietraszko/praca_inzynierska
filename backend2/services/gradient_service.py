import os
import numpy as np
from interfaces.gradient import GradientRequest, GradientResponse
from services.models_service import ModelInfo
import torch
import cv2


class GradientService:
    def __init__(self):
        pass

    def gradient(self, request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 models_service,
                 constants
                 ):
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target

            label_path = os.path.join(
                root_path, label
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
                image_path, model_info.input_shape
            )

            image_tensor = utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

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

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True
        )

    def gradient_squared(self, request: GradientRequest,
                         container,
                         datasets_service,
                         utilities_service,
                         models_service,
                         constants
                         ):
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.GRADIENT_SQUARED
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target

            label_path = os.path.join(
                root_path, label
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
                image_path, model_info.input_shape
            )

            image_tensor = utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

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

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True
        )

    def deconvolution(self, request: GradientRequest,
                      container,
                      datasets_service,
                      utilities_service,
                      models_service,
                      constants
                      ):
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.DECONVOLUTION
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

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

            label_path = os.path.join(
                root_path, label
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
                image_path, model_info.input_shape
            )

            image_tensor = utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            result = grad.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = utilities_service.aggregate_channels(
                    result, constants.AGGREGATION)

            cv2.imwrite(save_path, result)

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    def guided_backpropagation(self, request: GradientRequest,
                               container,
                               datasets_service,
                               utilities_service,
                               models_service,
                               constants
                               ):
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.GUIDED_BACKPROPAGATION
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

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

            label_path = os.path.join(
                root_path, label
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
                image_path, model_info.input_shape
            )

            image_tensor = utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

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

            cv2.imwrite(save_path, result)

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    def input_gradient(self, request: GradientRequest,
                       container,
                       datasets_service,
                       utilities_service,
                       models_service,
                       constants
                       ):
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.INPUT_GRADIENT
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target

            label_path = os.path.join(
                root_path, label
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
                image_path, model_info.input_shape
            )
            image_tensor = utilities_service.convert_image_to_tensor(
                image_np, grad=True
            )

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

            cv2.imwrite(save_path, result)

        return GradientResponse(
            success=True
        )
