import os
import cv2
import numpy as np
import torch
from interfaces.gradient import GradientRequest, GradientResponse


class CAMService:
    def __init__(self):
        pass

    def grad_cam(self,
                 request: GradientRequest,
                 container,
                 datasets_service,
                 utilities_service,
                 constants,
                 models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.GRAD_CAM
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

            activations = None
            gradients = None

            def forward_hook(module, i, o):
                nonlocal activations
                activations = o.clone().detach()

            def backward_hook(module, i, o):
                nonlocal gradients
                gradients = o[0].clone().detach()

            h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
            h2 = model_info.last_conv_layer.register_full_backward_hook(
                backward_hook)
            model_info.model.eval()
            image_tensor.requires_grad_(True)
            output = model_info.model(image_tensor)
            output[0, target].backward()
            h1.remove()
            h2.remove()

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = cam.squeeze().cpu().detach().numpy()

            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )

            cv2.imwrite(save_path, cam)

        return GradientResponse(
            success=True,
        )

    def grad_cam_plus_plus(self,
                           request: GradientRequest,
                           container,
                           datasets_service,
                           utilities_service,
                           constants,
                           models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.GRAD_CAM_PLUS_PLUS
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
        return GradientResponse(
            success=True,
        )

    def group_cam(self,
                  request: GradientRequest,
                  container,
                  datasets_service,
                  utilities_service,
                  constants,
                  models_service) -> GradientResponse:

        return GradientResponse(
            success=True,
        )

    def hires_cam(self,
                  request: GradientRequest,
                  container,
                  datasets_service,
                  utilities_service,
                  constants,
                  models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.HIRES_CAM
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

            activations = None
            gradients = None

            def forward_hook(module, i, o):
                nonlocal activations
                activations = o.clone().detach()

            def backward_hook(module, i, o):
                nonlocal gradients
                gradients = o[0].clone().detach()

            h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
            h2 = model_info.last_conv_layer.register_full_backward_hook(
                backward_hook)

            output = model_info.model(image_tensor)
            output[0, target].backward()
            h1.remove()
            h2.remove()

            weights = gradients

            cam = (activations * weights).sum(dim=1, keepdim=True)
            cam = cam.squeeze().cpu().detach().numpy()

            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )


            cv2.imwrite(save_path, cam)
        return GradientResponse(
            success=True,
        )

    def wgrad_cam(self,
                  request: GradientRequest,
                  container,
                  datasets_service,
                  utilities_service,
                  constants,
                  models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.WGRAD_CAM
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
        return GradientResponse(
            success=True,
        )

    def ablation_cam(self,
                     request: GradientRequest,
                     container,
                     datasets_service,
                     utilities_service,
                     constants,
                     models_service) -> GradientResponse:
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.ABLATION_CAM
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

            activations = None

            def forward_hook(module, inp, output):
                nonlocal activations
                activations = output

            h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)

            pred = model_info.model(image_tensor)
            y_c = pred[0, target].item()

            h1.remove()

            activations = activations.cpu().detach().numpy()
            weights = np.zeros(activations.shape[1], dtype=np.float32)

            for i in range(activations.shape[1]):
                def hook_zeros_k_map(module, inp, out):
                    modified_output = out.clone()
                    modified_output[:, i] = 0
                    return modified_output

                hook = model_info.last_conv_layer.register_forward_hook(
                    hook_zeros_k_map)

                pred_mod = model_info.model(image_tensor)
                y_k = pred_mod[0, target].item()

                weights[i] = abs(y_c - y_k) / y_c if y_c != 0 else 0

                hook.remove()

            activations = activations[0]
            ablationcam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                ablationcam += float(w) * activations[i, :, :]

            ablationcam = np.maximum(ablationcam, 0)
            ablationcam = cv2.normalize(
                ablationcam, None, 0, 255, cv2.NORM_MINMAX)

            ablationcam = cv2.resize(
                ablationcam, (model_info.input_shape[2], model_info.input_shape[1]))

            cv2.imwrite(save_path, ablationcam)
        return GradientResponse(
            success=True,
        )
