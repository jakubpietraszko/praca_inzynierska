import builtins
import contextlib
import os

import cv2
import torch
import torch.nn as nn

from interfaces.gradient import GradientRequest, GradientResponse
from services.config_service import XAIMethodConfig


class LRPService:
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
        self.logger: LoggerService = get_logger_service()

    async def lrp_epsilon(self, request: GradientRequest):
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.LRP_EPSILON,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LRP_EPSILON)

        self.logger.log(f"LRP Epsilon config: EPSILON={config.epsilon}, AGGREGATION={config.aggregation}")

        hooks = []
        original_inplace_states = {}
        stack_modules = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = []
        stack_outputs = []

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output

            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if r is None:
                    break

                if r.shape != current_output.shape:
                    try:
                        r = r.view_as(current_output)
                    except Exception:
                        r_sum = r.sum()
                        r = torch.zeros_like(current_input)
                        r += r_sum / r.numel()
                        continue

                z = current_module(current_input) + (config.epsilon or self.constants.DEFAULT_EPSILON)

                s = r / z
                c = torch.autograd.grad(
                    outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                )[0]

                r = c * current_input

            if r is None:
                break

            result = r.squeeze().cpu().detach().numpy()

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
                heatmap_type=self.constants.LRP_EPSILON,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(success=True)

    async def lrp_gamma(self, request: GradientRequest):
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.LRP_GAMMA,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LRP_GAMMA)

        self.logger.log(
            f"LRP Gamma config: EPSILON={config.epsilon}, AGGREGATION={config.aggregation}, GAMMA={config.gamma}"
        )

        hooks = []
        original_inplace_states = {}
        stack_modules = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = []
        stack_outputs = []

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output

            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if r is None:
                    break

                if r.shape != current_output.shape:
                    try:
                        r = r.view_as(current_output)
                    except Exception:
                        r_sum = r.sum()
                        r = torch.zeros_like(current_input)
                        r += r_sum / r.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_plus = torch.clamp(weight, min=0)
                    modified_weight = weight + (config.gamma or self.constants.DEFAULT_GAMMA) * weight_plus

                    if isinstance(current_module, nn.Conv2d):
                        z = torch.nn.functional.conv2d(
                            current_input,
                            modified_weight,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    elif isinstance(current_module, nn.Linear):
                        z = torch.nn.functional.linear(
                            current_input,
                            modified_weight,
                            bias=current_module.bias,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)

                else:
                    z = current_module(current_input) + (config.epsilon or self.constants.DEFAULT_EPSILON)

                if r is None:
                    break

                s = r / z
                c = torch.autograd.grad(
                    outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                )[0]

                r = c * current_input

            if r is None:
                continue

            result = r.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, (config.aggregation or self.constants.DEFAULT_AGGREGATION)
                )

            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=(config.epsilon or self.constants.DEFAULT_EPSILON)
            )

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.LRP_GAMMA,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(success=True)

    async def lrp_alpha_beta(self, request: GradientRequest):
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.LRP_ALPHA_BETA,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LRP_ALPHA_BETA)

        self.logger.log(
            f"LRP Alpha-Beta config: EPSILON={config.epsilon}, AGGREGATION={config.aggregation}, ALPHA={config.alpha}, BETA={config.beta}"
        )

        hooks = []
        original_inplace_states = {}
        stack_modules = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = []
        stack_outputs = []

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output

            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if r is None:
                    break

                if r.shape != current_output.shape:
                    try:
                        r = r.view_as(current_output)
                    except Exception:
                        r_sum = r.sum()
                        r = torch.zeros_like(current_input)
                        r += r_sum / r.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_pos = torch.clamp(weight, min=0)
                    weight_neg = torch.clamp(weight, max=0)

                    if isinstance(current_module, nn.Conv2d):
                        z_pos = torch.nn.functional.conv2d(
                            current_input,
                            weight_pos,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                        z_neg = torch.nn.functional.conv2d(
                            current_input,
                            weight_neg,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) - (config.epsilon or self.constants.DEFAULT_EPSILON)
                    else:
                        z_pos = torch.nn.functional.linear(
                            current_input,
                            weight_pos,
                            bias=current_module.bias,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                        z_neg = torch.nn.functional.linear(
                            current_input,
                            weight_neg,
                            bias=current_module.bias,
                        ) - (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s_pos = r / z_pos
                    s_neg = r / z_neg

                    c_pos = torch.autograd.grad(
                        outputs=z_pos, inputs=current_input, grad_outputs=s_pos, retain_graph=True, allow_unused=True
                    )[0]
                    c_neg = torch.autograd.grad(
                        outputs=z_neg, inputs=current_input, grad_outputs=s_neg, retain_graph=True, allow_unused=True
                    )[0]

                    r = (config.alpha or self.constants.DEFAULT_ALPHA) * c_pos * current_input - (
                        config.beta or self.constants.DEFAULT_BETA
                    ) * c_neg * current_input

                else:
                    z = current_module(current_input) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s = r / z
                    c = torch.autograd.grad(
                        outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                    )[0]
                    r = c * current_input

            if r is None:
                continue

            result = r.squeeze().cpu().detach().numpy()

            if len(result.shape) == 3 and result.shape[0] == 3:
                result = await self.utilities_service.aggregate_channels(
                    result, config.aggregation or self.constants.DEFAULT_AGGREGATION
                )

            result = await self.utilities_service.relu_normalize(
                result, model_info.input_shape, epsilon=(config.epsilon or self.constants.DEFAULT_EPSILON)
            )

            await self.utilities_service.save_image(save_path, result)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.LRP_ALPHA_BETA,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(success=True)

    async def lrp_zplus(self, request: GradientRequest):
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.LRP_ZPLUS,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LRP_ZPLUS)

        self.logger.log(f"LRP ZPlus config: EPSILON={config.epsilon}, AGGREGATION={config.aggregation}")

        hooks = []
        original_inplace_states = {}
        stack_modules = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = []
        stack_outputs = []

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output

            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if r is None:
                    break

                if r.shape != current_output.shape:
                    try:
                        r = r.view_as(current_output)
                    except Exception:
                        r_sum = r.sum()
                        r = torch.zeros_like(current_input)
                        r += r_sum / r.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_pos = torch.clamp(weight, min=0)

                    if isinstance(current_module, nn.Conv2d):
                        z = torch.nn.functional.conv2d(
                            current_input,
                            weight_pos,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    else:
                        z = torch.nn.functional.linear(
                            current_input,
                            weight_pos,
                            bias=current_module.bias,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s = r / z
                    c = torch.autograd.grad(
                        outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                    )[0]
                    r = c * current_input
                else:
                    z = current_module(current_input) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s = r / z
                    c = torch.autograd.grad(
                        outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                    )[0]
                    r = c * current_input

            if r is None:
                continue

            result = r.squeeze().cpu().detach().numpy()

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
                heatmap_type=self.constants.LRP_ZPLUS,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(success=True)

    async def lrp_w2(self, request: GradientRequest):
        root_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.LRP_W2,
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await self.models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LRP_W2)

        self.logger.log(f"LRP W2 config: EPSILON={config.epsilon}, AGGREGATION={config.aggregation}")

        hooks = []
        original_inplace_states = {}
        stack_modules = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = []
        stack_outputs = []

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output

            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if r is None:
                    break

                if r.shape != current_output.shape:
                    try:
                        r = r.view_as(current_output)
                    except Exception:
                        r_sum = 1e18
                        r = torch.zeros_like(current_input)
                        r += r_sum / r.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight2 = weight * weight

                    if isinstance(current_module, nn.Conv2d):
                        denom = weight2.sum(dim=1, keepdim=True) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                        norm_weight = weight2 / denom
                        z = torch.nn.functional.conv2d(
                            current_input,
                            norm_weight,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    else:
                        denom = weight2.sum(dim=0, keepdim=True) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                        norm_weight = weight2 / denom
                        z = torch.nn.functional.linear(
                            current_input,
                            norm_weight,
                            bias=current_module.bias,
                        ) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s = r / z
                    c = torch.autograd.grad(
                        outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                    )[0]
                    r = c * current_input

                else:
                    z = current_module(current_input) + (config.epsilon or self.constants.DEFAULT_EPSILON)
                    s = r / z
                    c = torch.autograd.grad(
                        outputs=z, inputs=current_input, grad_outputs=s, retain_graph=True, allow_unused=True
                    )[0]
                    r = c * current_input

            if r is None:
                continue

            result = r.squeeze().cpu().detach().numpy()

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
                heatmap_type=self.constants.LRP_W2,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(success=True)
