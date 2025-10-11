import os
from interfaces.gradient import GradientRequest, GradientResponse
import torch
import torch.nn as nn
import cv2


class LRPService:
    def __init__(self):
        from services.container import get_config_service, get_websocket_service
        from services.container import get_constants, Constants
        self.config_service = get_config_service()
        self.websocket_service = get_websocket_service()
        self.constants: Constants = get_constants()

    async def lrp_epsilon(self, request: GradientRequest,
                          container,
                          datasets_service,
                          utilities_service,
                          models_service,
                          constants):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_EPSILON
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config = await self.config_service.get_method_config('lrp_epsilon')
        EPSILON = config.epsilon or 1e-6
        AGGREGATION = config.aggregation or 'max'

        hooks = list()
        original_inplace_states = dict()
        stack_modules = list()
        R = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = list()
        stack_outputs = list()

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(
                    input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output
            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(
                    forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await utilities_service.skip_forward(save_path):
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

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            R = torch.zeros_like(output)
            R[0, target] = 1.0
            R = R.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if R.shape != current_output.shape:
                    try:
                        R = R.view_as(current_output)
                    except:
                        R_sum = R.sum()
                        R = torch.zeros_like(current_input)
                        R += R_sum / R.numel()
                        continue

                z = current_module(current_input) + EPSILON

                s = R / z
                c = torch.autograd.grad(
                    outputs=z,
                    inputs=current_input,
                    grad_outputs=s,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                R = c * current_input

            result = R.squeeze().cpu().detach().numpy()

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
                    heatmap_type="lrp_epsilon",
                )
            except:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def lrp_gamma(self, request: GradientRequest,
                        container,
                        datasets_service,
                        utilities_service,
                        models_service,
                        constants):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_GAMMA
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config = await self.config_service.get_method_config('lrp_gamma')
        EPSILON = config.epsilon or 1e-6
        AGGREGATION = config.aggregation or 'max'
        GAMMA = config.gamma or 0.25

        hooks = list()
        original_inplace_states = dict()
        stack_modules = list()
        R = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = list()
        stack_outputs = list()

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(
                    input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output
            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(
                    forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await utilities_service.skip_forward(save_path):
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

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            R = torch.zeros_like(output)
            R[0, target] = 1.0
            R = R.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if R.shape != current_output.shape:
                    try:
                        R = R.view_as(current_output)
                    except:
                        R_sum = R.sum()
                        R = torch.zeros_like(current_input)
                        R += R_sum / R.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_plus = torch.clamp(weight, min=0)
                    modified_weight = weight + GAMMA * weight_plus

                    if isinstance(current_module, nn.Conv2d):
                        z = torch.nn.functional.conv2d(
                            current_input, modified_weight,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + EPSILON
                    elif isinstance(current_module, nn.Linear):
                        z = torch.nn.functional.linear(
                            current_input, modified_weight,
                            bias=current_module.bias,
                        ) + EPSILON

                else:
                    z = current_module(current_input) + EPSILON

                s = R / z
                c = torch.autograd.grad(
                    outputs=z,
                    inputs=current_input,
                    grad_outputs=s,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                R = c * current_input

            result = R.squeeze().cpu().detach().numpy()

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
                    heatmap_type="lrp_gamma",
                )
            except:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def lrp_alpha_beta(self, request: GradientRequest,
                             container,
                             datasets_service,
                             utilities_service,
                             models_service,
                             constants):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_ALPHA_BETA
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config = await self.config_service.get_method_config('lrp_alpha_beta')
        EPSILON = config.epsilon or 1e-6
        AGGREGATION = config.aggregation or 'max'
        ALPHA = config.alpha or 1.5
        BETA = config.beta or 0.5

        hooks = list()
        original_inplace_states = dict()
        stack_modules = list()
        R = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = list()
        stack_outputs = list()

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(
                    input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output
            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(
                    forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await utilities_service.skip_forward(save_path):
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

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            R = torch.zeros_like(output)
            R[0, target] = 1.0
            R = R.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if R.shape != current_output.shape:
                    try:
                        R = R.view_as(current_output)
                    except:
                        R_sum = R.sum()
                        R = torch.zeros_like(current_input)
                        R += R_sum / R.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_pos = torch.clamp(weight, min=0)
                    weight_neg = torch.clamp(weight, max=0)

                    if isinstance(current_module, nn.Conv2d):
                        z_pos = torch.nn.functional.conv2d(
                            current_input, weight_pos,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + EPSILON
                        z_neg = torch.nn.functional.conv2d(
                            current_input, weight_neg,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) - EPSILON
                    else:
                        z_pos = torch.nn.functional.linear(
                            current_input, weight_pos,
                            bias=current_module.bias,
                        ) + EPSILON
                        z_neg = torch.nn.functional.linear(
                            current_input, weight_neg,
                            bias=current_module.bias,
                        ) - EPSILON
                    s_pos = R / z_pos
                    s_neg = R / z_neg

                    c_pos = torch.autograd.grad(
                        outputs=z_pos,
                        inputs=current_input,
                        grad_outputs=s_pos,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    c_neg = torch.autograd.grad(
                        outputs=z_neg,
                        inputs=current_input,
                        grad_outputs=s_neg,
                        retain_graph=True,
                        allow_unused=True
                    )[0]

                    R = ALPHA * c_pos * current_input - BETA * c_neg * current_input

                else:
                    z = current_module(current_input) + EPSILON
                    s = R / z
                    c = torch.autograd.grad(
                        outputs=z,
                        inputs=current_input,
                        grad_outputs=s,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    R = c * current_input

            result = R.squeeze().cpu().detach().numpy()

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
                    heatmap_type="lrp_alpha_beta",
                )
            except:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def lrp_zplus(self, request: GradientRequest,
                        container,
                        datasets_service,
                        utilities_service,
                        models_service,
                        constants):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_ZPLUS
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config = await self.config_service.get_method_config('lrp_zplus')
        EPSILON = config.epsilon or 1e-6
        AGGREGATION = config.aggregation or 'max'

        hooks = list()
        original_inplace_states = dict()
        stack_modules = list()
        R = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = list()
        stack_outputs = list()

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(
                    input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output
            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(
                    forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await utilities_service.skip_forward(save_path):
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

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            R = torch.zeros_like(output)
            R[0, target] = 1.0
            R = R.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if R.shape != current_output.shape:
                    try:
                        R = R.view_as(current_output)
                    except:
                        R_sum = R.sum()
                        R = torch.zeros_like(current_input)
                        R += R_sum / R.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight_pos = torch.clamp(weight, min=0)

                    if isinstance(current_module, nn.Conv2d):
                        z = torch.nn.functional.conv2d(
                            current_input, weight_pos,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + EPSILON
                    else:
                        z = torch.nn.functional.linear(
                            current_input, weight_pos,
                            bias=current_module.bias,
                        ) + EPSILON
                    s = R / z
                    c = torch.autograd.grad(
                        outputs=z,
                        inputs=current_input,
                        grad_outputs=s,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    R = c * current_input
                else:
                    z = current_module(current_input) + EPSILON
                    s = R / z
                    c = torch.autograd.grad(
                        outputs=z,
                        inputs=current_input,
                        grad_outputs=s,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    R = c * current_input

            result = R.squeeze().cpu().detach().numpy()

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
                    heatmap_type="lrp_zplus",
                )
            except:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )

    async def lrp_w2(self, request: GradientRequest,
                     container,
                     datasets_service,
                     utilities_service,
                     models_service,
                     constants):
        root_path = os.path.join(
            container.models_datasets_path,
            await utilities_service.models_datasets_path(
                request.model, request.dataset),
            constants.LRP_W2
        )
        os.makedirs(root_path, exist_ok=True)

        model_info = await models_service.load_model(request.model)
        model = model_info.model
        model.eval()

        config = await self.config_service.get_method_config('lrp_w2')
        EPSILON = config.epsilon or 1e-6
        AGGREGATION = config.aggregation or 'max'

        hooks = list()
        original_inplace_states = dict()
        stack_modules = list()
        R = None

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        stack_inputs = list()
        stack_outputs = list()

        def forward_hook(name):
            def hook(module, input, output):
                nonlocal stack_modules, stack_inputs, stack_outputs
                stack_modules.append(module)
                stack_inputs.append(
                    input[0] if isinstance(input, tuple) else input)
                stack_outputs.append(output)
                return output
            return hook

        for image_info in request.images:
            for name, module in model_info.model.named_modules():
                if len(list(module.children())) > 0 or module is model_info.model:
                    continue
                forward_hook_handle = module.register_forward_hook(
                    forward_hook(name))
                hooks.append(forward_hook_handle)
            label = image_info.label
            filename = image_info.filename
            target = image_info.target

            if target is None:
                target = await utilities_service.fill_target(model_info, label)

            label_path = os.path.join(root_path, label)
            os.makedirs(label_path, exist_ok=True)
            save_path = os.path.join(
                label_path,
                await utilities_service.add_png_extension(filename)
            )

            if self.constants.CACHE_ENABLED and await utilities_service.skip_forward(save_path):
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

            stack_modules.clear()
            stack_inputs.clear()
            stack_outputs.clear()

            output = model(image_tensor)
            for hook in hooks:
                hook.remove()
            R = torch.zeros_like(output)
            R[0, target] = 1.0
            R = R.squeeze()

            while len(stack_modules) > 0:
                current_module = stack_modules.pop()
                current_input = stack_inputs.pop()
                current_output = stack_outputs.pop()

                if R.shape != current_output.shape:
                    try:
                        R = R.view_as(current_output)
                    except:
                        R_sum = 1e18
                        R = torch.zeros_like(current_input)
                        R += R_sum / R.numel()
                        continue

                if isinstance(current_module, (nn.Conv2d, nn.Linear)):
                    weight = current_module.weight.clone()
                    weight2 = weight * weight

                    if isinstance(current_module, nn.Conv2d):
                        denom = weight2.sum(dim=1, keepdim=True) + EPSILON
                        norm_weight = weight2 / denom
                        z = torch.nn.functional.conv2d(
                            current_input, norm_weight,
                            bias=current_module.bias,
                            stride=current_module.stride,
                            padding=current_module.padding,
                            dilation=current_module.dilation,
                            groups=current_module.groups,
                        ) + EPSILON
                    else:
                        denom = weight2.sum(dim=0, keepdim=True) + EPSILON
                        norm_weight = weight2 / denom
                        z = torch.nn.functional.linear(
                            current_input, norm_weight,
                            bias=current_module.bias,
                        ) + EPSILON
                    s = R / z
                    c = torch.autograd.grad(
                        outputs=z,
                        inputs=current_input,
                        grad_outputs=s,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    R = c * current_input

                else:
                    z = current_module(current_input) + EPSILON
                    s = R / z
                    c = torch.autograd.grad(
                        outputs=z,
                        inputs=current_input,
                        grad_outputs=s,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    R = c * current_input

            result = R.squeeze().cpu().detach().numpy()

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
                    heatmap_type="lrp_w2",
                )
            except:
                pass

        for hook in hooks:
            hook.remove()

        for name, module in model_info.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = original_inplace_states[name]

        return GradientResponse(
            success=True
        )
