import os

import torch
import torch.nn as nn

from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


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

    async def lrp_epsilon(self, request: HeatmapRequest) -> HeatmapResponse:
        return await self.__generic_lrp(request, self.constants.LRP_EPSILON)

    async def lrp_gamma(self, request: HeatmapRequest) -> HeatmapResponse:
        return await self.__generic_lrp(request, self.constants.LRP_GAMMA)

    async def lrp_alpha_beta(self, request: HeatmapRequest) -> HeatmapResponse:
        return await self.__generic_lrp(request, self.constants.LRP_ALPHA_BETA)

    async def lrp_zplus(self, request: HeatmapRequest) -> HeatmapResponse:
        return await self.__generic_lrp(request, self.constants.LRP_ZPLUS)

    async def lrp_w2(self, request: HeatmapRequest) -> HeatmapResponse:
        return await self.__generic_lrp(request, self.constants.LRP_W2)

    async def __generic_lrp(self, request: HeatmapRequest, xai_method: str) -> HeatmapResponse:
        root_path: str = await self.utilities_service.make_root_path(request, xai_method)

        model_info: ModelInfo = await self.models_service.load_model(request.model)
        model: nn.Module = model_info.model
        model.eval()

        config: XAIMethodConfig = await self.config_service.get_method_config(xai_method)

        self.logger.log_config(config=config, xai_method_name=xai_method)

        hooks: list[torch.utils.hooks.RemovableHandle] = []
        original_inplace_states: dict[str, bool] = {}
        stack_modules: list[nn.Module] = []
        r: torch.Tensor | None = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs: list[torch.Tensor] = []
        stack_outputs: list[torch.Tensor] = []

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

            label_path: str = await self.utilities_service.make_label_path(root_path, label)

            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)
            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output: torch.Tensor = model(image_tensor)
            for hook in hooks:
                hook.remove()
            r = torch.zeros_like(output)
            r[0, target] = 1.0
            r = r.squeeze()

            if xai_method == self.constants.LRP_EPSILON:
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
            elif xai_method == self.constants.LRP_GAMMA:
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
            elif xai_method == self.constants.LRP_ALPHA_BETA:
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
                            outputs=z_pos,
                            inputs=current_input,
                            grad_outputs=s_pos,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        c_neg = torch.autograd.grad(
                            outputs=z_neg,
                            inputs=current_input,
                            grad_outputs=s_neg,
                            retain_graph=True,
                            allow_unused=True,
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
            elif xai_method == self.constants.LRP_ZPLUS:
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
            elif xai_method == self.constants.LRP_W2:
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
                            denom = weight2.sum(dim=1, keepdim=True) + (
                                config.epsilon or self.constants.DEFAULT_EPSILON
                            )
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
                            denom = weight2.sum(dim=0, keepdim=True) + (
                                config.epsilon or self.constants.DEFAULT_EPSILON
                            )
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
            else:
                self.logger.log(f"Unknown LRP method: {xai_method}")
                continue

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
                heatmap_type=xai_method,
            )

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return HeatmapResponse(success=True)
