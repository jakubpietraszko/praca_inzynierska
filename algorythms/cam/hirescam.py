import torch
import numpy as np

def hirescam(
        image: torch.Tensor,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        target_class: int = None,
        normalize: bool = True
) -> np.ndarray:

    activations = None
    gradients = None

    def forward_hook(module, i, o):
        nonlocal activations
        activations = o

    def backward_hook(module, i, o):
        nonlocal gradients
        gradients = o[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    image = image.unsqueeze(0)
    model.eval()
    output = model(image)
    output[0, target_class].backward()

    h1.remove()
    h2.remove()

    activations = activations
    gradients = gradients

    weights = gradients

    cam = (activations * weights).sum(dim=1, keepdim=True)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)

    if normalize:
        cam -= np.min(cam)
        cam /= np.max(cam)

    cam *= 255.0
    cam = cam.astype(np.uint8)
    
    return cam