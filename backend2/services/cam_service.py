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
            # activations (1, b, h, w)
            # gradients (1, b, h, w)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            # weights (1, b, 1, 1)
            # normalize weights
            weights = weights / (torch.sum(weights) + constants.EPSILON)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            # cam (1, 1, h, w)
            cam = cam.squeeze().cpu().detach().numpy()
            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )
            # cam (H, W)

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

            grads = gradients
            grads_2 = grads.pow(2)
            grads_3 = grads.pow(3)

            # activations (1, b, h, w)
            # grads (1, b, h, w)
            # grads_2 (1, b, h, w)
            # grads_3 (1, b, h, w)
            spatial_sums = (
                activations * grads_3).sum(dim=(2, 3), keepdim=True)
            # spatial_sums (1, b, 1, 1)
            epsilon = constants.EPSILON
            alphas = grads_2 / (2 * grads_2 + spatial_sums + epsilon)
            # alphas (1, b, h, w)
            relu_grads = torch.relu(grads)
            weights = (alphas * relu_grads).sum(dim=(2, 3), keepdim=True)
            # weights (1, b, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            # cam (1, 1, h, w)
            cam = cam.squeeze().cpu().detach().numpy()
            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )
            # cam (H, W)
            cv2.imwrite(save_path, cam)

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
        root_path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.GROUP_CAM
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

        G = 8
        ksize = 11
        sigma = 1.0

        for image in request.images:
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
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

            image_np_for_blur = image_tensor.squeeze().cpu().detach().numpy()
            if image_np_for_blur.ndim == 3:
                image_np_for_blur = np.transpose(image_np_for_blur, (1, 2, 0))

            blurred_np = cv2.GaussianBlur(
                image_np_for_blur, (ksize, ksize), sigma)

            if blurred_np.ndim == 3:
                blurred_np = np.transpose(blurred_np, (2, 0, 1))
            blurred_image = torch.from_numpy(blurred_np).unsqueeze(0).float()
            if image_tensor.device.type == 'cuda':
                blurred_image = blurred_image.cuda()

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

            image_tensor.requires_grad_(True)
            output = model_info.model(image_tensor)
            output[0, target].backward()
            h1.remove()
            h2.remove()

            # activations (1, b, h, w)
            # gradients (1, b, h, w)

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            # weights (1, b, 1, 1)

            K = activations.shape[1]
            g = K // G

            group_cam_result = torch.zeros_like(image_tensor)

            with torch.no_grad():
                baseline_output = model_info.model(blurred_image)
                baseline_confidence = torch.softmax(baseline_output, dim=1)[
                    0, target].item()

            for l in range(G):
                start_idx = l * g
                end_idx = min((l + 1) * g, K)

                group_activations = activations[:, start_idx:end_idx, :, :]
                group_weights = weights[:, start_idx:end_idx, :, :]

                M_l = torch.relu(
                    (group_weights * group_activations).sum(dim=1, keepdim=True))

                M_l_processed = utilities_service.relu_normalize(
                    M_l.squeeze().cpu().detach().numpy(),
                    model_info.input_shape,
                    constants.EPSILON
                )

                M_l_prime = torch.from_numpy(
                    M_l_processed).unsqueeze(0).unsqueeze(0).float()
                M_l_prime = torch.nn.functional.interpolate(
                    M_l_prime,
                    size=(model_info.input_shape[1],
                          model_info.input_shape[2]),
                    mode='bilinear',
                    align_corners=False
                )

                M_l_prime = (M_l_prime - M_l_prime.min()) / \
                    (M_l_prime.max() - M_l_prime.min() + constants.EPSILON)

                I_l_prime = image_tensor * M_l_prime + \
                    blurred_image * (1 - M_l_prime)

                with torch.no_grad():
                    perturbed_output = model_info.model(I_l_prime)
                    perturbed_confidence = torch.softmax(perturbed_output, dim=1)[
                        0, target].item()

                alpha_l = perturbed_confidence - baseline_confidence

                group_cam_result += alpha_l * M_l_prime

            group_cam_final = torch.relu(group_cam_result)
            group_cam_final = group_cam_final.squeeze().cpu().detach().numpy()

            if group_cam_final.ndim == 3:
                group_cam_final = np.mean(group_cam_final, axis=0)

            group_cam_final = cv2.normalize(
                group_cam_final, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imwrite(save_path, group_cam_final)

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

            # activations (1, b, h, w)
            # gradients (1, b, h, w)

            weights = gradients
            # weights (1, b, h, w)

            cam = (activations * weights).sum(dim=1, keepdim=True)
            # cam (1, 1, h, w)
            cam = cam.squeeze().cpu().detach().numpy()
            # cam (h, w)

            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )
            # cam (H, W)

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

            image_tensor.requires_grad_(True)
            output = model_info.model(image_tensor)

            output[0, target].backward()
            h1.remove()
            h2.remove()

            # activations (1, b, h, w)
            # gradients (1, b, h, w)

            K = activations.shape[1]

            weights = torch.zeros(K, device=activations.device)
            # weights (b,)

            for k in range(K):
                A_k = activations[0, k, :, :]
                G_k = gradients[0, k, :, :]
                weights[k] = torch.mean(A_k * G_k)

            weights = weights / (torch.sum(weights) + constants.EPSILON)

            weights = weights.unsqueeze(-1).unsqueeze(-1)
            # weights (b, 1, 1)
            cam = torch.sum(weights * activations[0], dim=0)
            # cam (h, w)

            cam = torch.relu(cam)

            cam = cam.cpu().detach().numpy()

            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )
            # cam (H, W)

            cv2.imwrite(save_path, cam)

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

            activations = None

            def forward_hook(module, inp, output):
                nonlocal activations
                activations = output

            h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)

            pred = model_info.model(image_tensor)
            y_c = pred[0, target].item()

            h1.remove()

            activations = activations.cpu().detach().numpy()
            # activations (1, b, h, w)
            weights = np.zeros(activations.shape[1], dtype=np.float32)
            # weights (b,)

            for i in range(activations.shape[1]):
                def hook_zeros_k_map(module, inp, out):
                    modified_output = out.clone()
                    modified_output[:, i] = 0
                    return modified_output

                hook = model_info.last_conv_layer.register_forward_hook(
                    hook_zeros_k_map)

                pred_mod = model_info.model(image_tensor)
                y_k = pred_mod[0, target].item()

                weights[i] = abs((y_c - y_k) / y_c) if y_c != 0 else 0

                hook.remove()

            activations = activations[0]
            # activations (b, h, w)
            ablationcam = np.zeros(activations.shape[1:], dtype=np.float32)
            # ablationcam (h, w)

            # normalize weights
            weights = weights / (np.sum(weights) + constants.EPSILON)

            for i, w in enumerate(weights):
                ablationcam += float(w) * activations[i, :, :]

            ablationcam = np.maximum(ablationcam, 0)
            ablationcam = cv2.normalize(
                ablationcam, None, 0, 255, cv2.NORM_MINMAX)

            ablationcam = cv2.resize(
                ablationcam, (model_info.input_shape[2], model_info.input_shape[1]))
            # ablationcam (H, W)

            cv2.imwrite(save_path, ablationcam)
        return GradientResponse(
            success=True,
        )

    def xgrad_cam(self,
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
            constants.XGRAD_CAM
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = models_service.load_model(request.model)

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

            image_tensor.requires_grad_(True)
            output = model_info.model(image_tensor)

            output[0, target].backward()
            h1.remove()
            h2.remove()

            # activations (1, b, h, w)
            # gradients (1, b, h, w)

            activations_x_gradient = activations * gradients
            # activations_x_gradient (1, b, h, w)

            weights = activations_x_gradient.mean(dim=(2, 3), keepdim=True)
            # weights (1, b, 1, 1)

            cam = (weights * activations).sum(dim=1, keepdim=True)
            # cam (1, 1, h, w)

            cam = torch.relu(cam)

            cam = cam.squeeze().cpu().detach().numpy()

            cam = utilities_service.relu_normalize(
                cam, model_info.input_shape, constants.EPSILON
            )
            # cam (H, W)

            cv2.imwrite(save_path, cam)

        return GradientResponse(
            success=True,
        )
