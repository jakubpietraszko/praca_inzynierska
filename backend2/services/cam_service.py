import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


class CAMService:
    def __init__(self) -> None:
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
        self.container: ServiceContainer = get_container()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.models_service: ModelsService = get_models_service()
        self.logger_service: LoggerService = get_logger_service()

    async def grad_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.GRAD_CAM, ref)

    async def grad_cam_plus_plus(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.GRAD_CAM_PLUS_PLUS, ref)

    async def group_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.GROUP_CAM, ref)

    async def hires_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.HIRES_CAM, ref)

    async def wgrad_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.WGRAD_CAM, ref)

    async def ablation_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.ABLATION_CAM, ref)

    async def xgrad_cam(self, request: HeatmapRequest, ref: None | str = None) -> HeatmapResponse:
        return await self.__generic_cam(request, self.constants.XGRAD_CAM, ref)

    async def __generic_cam(self, request: HeatmapRequest, xai_method: str, ref: None | str = None) -> HeatmapResponse:
        root_path: str = await self.utilities_service.make_root_path(request, xai_method)

        model_info: ModelInfo = await self.models_service.load_model(request.model)
        model: nn.Module = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(xai_method)

        self.logger_service.log_config(config, xai_method)

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

            activations: torch.Tensor | None = None
            gradients: torch.Tensor | None = None

            def forward_hook(module, i, o):
                nonlocal activations
                activations = o.clone().detach()

            def backward_hook(module, i, o):
                nonlocal gradients
                gradients = o[0].clone().detach()

            h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
            h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)

            model_info.model.eval()
            image_tensor.requires_grad_(True)
            output = model_info.model(image_tensor)
            output[0, target].backward()
            h1.remove()
            h2.remove()
            # activations (1, b, h, w)
            # gradients (1, b, h, w)
            if activations is None or gradients is None:
                raise RuntimeError("Failed to obtain activations or gradients from the model.")
            assert activations.shape == gradients.shape, "Activations and gradients must have the same shape."
            assert activations.shape[0] == 1, "Batch size must be 1."
            assert activations.ndim == 4, "Activations must be 4-dimensional."

            if xai_method == self.constants.GRAD_CAM:
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                weights = torch.nn.functional.relu(weights)
                weights = weights / (torch.sum(weights) + (config.epsilon or self.constants.DEFAULT_EPSILON))
                cam = (weights * activations).sum(dim=1, keepdim=True)
            elif xai_method == self.constants.GRAD_CAM_PLUS_PLUS:
                grads = gradients
                grads_2 = grads.pow(2)
                grads_3 = grads.pow(3)
                spatial_sums = (activations * grads_3).sum(dim=(2, 3), keepdim=True)
                alphas = grads_2 / (2 * grads_2 + spatial_sums + (config.epsilon or self.constants.DEFAULT_EPSILON))
                relu_grads = torch.relu(grads)
                weights = (alphas * relu_grads).sum(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
            elif xai_method == self.constants.HIRES_CAM:
                weights = gradients
                cam = (activations * weights).sum(dim=1, keepdim=True)
            elif xai_method == self.constants.WGRAD_CAM:
                K = activations.shape[1]
                weights = torch.zeros(K, device=activations.device)
                for k in range(K):
                    A_k = activations[0, k, :, :]
                    G_k = gradients[0, k, :, :]
                    weights[k] = torch.mean(A_k * G_k)
                weights = weights / (torch.sum(weights) + (config.epsilon or self.constants.DEFAULT_EPSILON))
                weights = weights.unsqueeze(-1).unsqueeze(-1)
                cam = torch.sum(weights * activations[0], dim=0, keepdim=True)
            elif xai_method == self.constants.XGRAD_CAM:
                activations_x_gradient = activations * gradients
                weights = activations_x_gradient.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
                cam = torch.relu(cam)
            elif xai_method == self.constants.ABLATION_CAM:
                y_c = output[0, target].item()
                weights = torch.zeros(activations.shape[1], dtype=torch.float32, device=activations.device)
                for i in range(activations.shape[1]):

                    def hook_zeros_k_map(module, inp, out):
                        modified_output = out.clone()
                        modified_output[:, i] = 0
                        return modified_output

                    hook = model_info.last_conv_layer.register_forward_hook(hook_zeros_k_map)

                    pred_mod = model_info.model(image_tensor)
                    y_k = pred_mod[0, target].item()

                    weights[i] = abs((y_c - y_k) / y_c) if y_c != 0 else 0

                    hook.remove()

                weights = weights / (torch.sum(weights) + (config.epsilon or self.constants.DEFAULT_EPSILON))
                weights = weights.view(1, -1, 1, 1)
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                cam = torch.relu(cam)
            elif xai_method == self.constants.GROUP_CAM:
                image_np_for_blur = image_tensor.squeeze().cpu().detach().numpy()
                if image_np_for_blur.ndim == 3:
                    image_np_for_blur = np.transpose(image_np_for_blur, (1, 2, 0))
                blurred_np = cv2.GaussianBlur(
                    image_np_for_blur,
                    (config.k_size or self.constants.DEFAULT_KSIZE, config.k_size or self.constants.DEFAULT_KSIZE),
                    config.sigma or self.constants.DEFAULT_SIGMA,
                )
                if blurred_np.ndim == 3:
                    blurred_np = np.transpose(blurred_np, (2, 0, 1))
                blurred_image = torch.from_numpy(blurred_np).unsqueeze(0).float()
                if image_tensor.device.type == "cuda":
                    blurred_image = blurred_image.cuda()
                activations: torch.Tensor | None = None
                gradients: torch.Tensor | None = None

                def forward_hook(module, i, o):
                    nonlocal activations
                    activations = o.clone().detach()

                def backward_hook(module, i, o):
                    nonlocal gradients
                    gradients = o[0].clone().detach()

                h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
                h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)

                image_tensor.requires_grad_(True)
                output = model_info.model(image_tensor)
                output[0, target].backward()
                h1.remove()
                h2.remove()

                if activations is None or gradients is None:
                    raise RuntimeError("Failed to obtain activations or gradients from the model.")

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                K = activations.shape[1]
                g = K // (config.group_size or self.constants.DEFAULT_GROUP_SIZE)
                group_cam_result = torch.zeros_like(image_tensor)

                with torch.no_grad():
                    baseline_output = model_info.model(blurred_image)
                    baseline_confidence = torch.softmax(baseline_output, dim=1)[0, target].item()

                for l in range(config.group_size or self.constants.DEFAULT_GROUP_SIZE):
                    start_idx = l * g
                    end_idx = min((l + 1) * g, K)

                    if activations is None or weights is None:
                        raise RuntimeError("Failed to obtain activations or weights from the model.")

                    group_activations = activations[:, start_idx:end_idx, :, :]
                    group_weights = weights[:, start_idx:end_idx, :, :]

                    M_l = torch.relu((group_weights * group_activations).sum(dim=1, keepdim=True))

                    M_l_processed = await self.utilities_service.relu_normalize(
                        M_l.squeeze().cpu().detach().numpy(),
                        model_info.input_shape,
                        config.epsilon or self.constants.DEFAULT_EPSILON,
                    )

                    M_l_prime = torch.from_numpy(M_l_processed).unsqueeze(0).unsqueeze(0).float()
                    M_l_prime = torch.nn.functional.interpolate(
                        M_l_prime,
                        size=(model_info.input_shape[1], model_info.input_shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    )

                    M_l_prime = (M_l_prime - M_l_prime.min()) / (
                        M_l_prime.max() - M_l_prime.min() + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    )

                    I_l_prime = image_tensor * M_l_prime + blurred_image * (1 - M_l_prime)

                    with torch.no_grad():
                        perturbed_output = model_info.model(I_l_prime)
                        perturbed_confidence = torch.softmax(perturbed_output, dim=1)[0, target].item()

                    alpha_l = perturbed_confidence - baseline_confidence

                    group_cam_result += alpha_l * M_l_prime
                cam = torch.relu(group_cam_result)

            else:
                raise NotImplementedError(f"XAI method {xai_method} is not implemented in generic CAM.")
            cam = cam.squeeze().cpu().detach().numpy()

            if cam.ndim == 3:
                cam = np.mean(cam, axis=0)
            cam = await self.utilities_service.relu_normalize(
                cam, model_info.input_shape, config.epsilon or self.constants.DEFAULT_EPSILON
            )

            await self.utilities_service.save_image(save_path, cam)


            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=xai_method,
            )

        return HeatmapResponse(
            success=True,
        )
