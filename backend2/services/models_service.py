import os
from dataclasses import dataclass
import torch.nn as nn
import torchvision.transforms as transforms
import importlib
import torch

@dataclass
class ModelInfo:
    name: str
    model: nn.Module
    last_conv_layer: nn.Module
    transform: transforms.Compose
    label_to_idx: dict
    input_shape: tuple

class ModelsService:
    def __init__(self, models_path: str):
        self.models_path = models_path

    def list_available_models(self) -> list[str]:
        models = os.listdir(self.models_path)
        return models
    
    def load_model(self, model_name: str) -> ModelInfo:
        path = os.path.join(self.models_path, model_name, 'model.py')
        spec = importlib.util.spec_from_file_location('loadModel', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model, last_conv_layer, transform, label_to_idx, input_shape = getattr(module, 'loadModel')()
        model_info = ModelInfo(
            name=model_name,
            model=model,
            last_conv_layer=last_conv_layer,
            transform=transform,
            label_to_idx=label_to_idx,
            input_shape=input_shape
        )

        return model_info

    def get_model_layers(self, model_name: str) -> list[str]:
        model_info = self.load_model(model_name)
        dummy_input = torch.randn(1, *model_info.input_shape)
        layers = list()
        hooks = list()

        def hook_fn(name):
            def hook(module, input, output):
                layers.append((name, module))
            return hook
        
        for name, module in model_info.model.named_modules():
            if name:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        model_info.model.eval()

        with torch.no_grad():
            model_info.model(dummy_input)

        for hook in hooks:
            hook.remove()

        result = list()

        for name, layer in layers:
            if ':' not in str(layer):
                result.append(str(layer))

        return result
