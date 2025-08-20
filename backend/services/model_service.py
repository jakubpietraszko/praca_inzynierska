import os
import importlib.util
import torch
from typing import Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str
    model: torch.nn.Module
    last_conv: torch.nn.Module
    transform: Any
    idx_to_label: dict
    shape: tuple


class ModelService:
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self._current_model: Optional[ModelInfo] = None
    
    def load_model(self, model_name: str) -> ModelInfo:
        path = os.path.join(self.models_path, model_name, 'model.py')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            spec = importlib.util.spec_from_file_location('loadModel', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'loadModel'):
                raise AttributeError(f"Model file {path} does not have 'loadModel' function")
            
            model, last_conv, transform, idx_to_label, shape = getattr(module, 'loadModel')()
            
            model_info = ModelInfo(
                name=model_name,
                model=model,
                last_conv=last_conv,
                transform=transform,
                idx_to_label=idx_to_label,
                shape=shape
            )
            
            return model_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def set_current_model(self, model_info: ModelInfo) -> None:
        self._current_model = model_info
    
    def get_current_model(self) -> Optional[ModelInfo]:
        return self._current_model
    
    def list_available_models(self) -> list[str]:
        if not os.path.exists(self.models_path):
            return []
        
        models = []
        for item in os.listdir(self.models_path):
            model_path = os.path.join(self.models_path, item)
            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, 'model.py')
                if os.path.exists(model_file):
                    models.append(item)
        
        return models
    
    def get_model_layers(self, model_name: str) -> list[str]:
        model_info = self.load_model(model_name)
        
        dummy_input = torch.randn(1, *model_info.shape)
        layers = []
        hooks = []

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
        
        return [str(layer) for _, layer in layers]
