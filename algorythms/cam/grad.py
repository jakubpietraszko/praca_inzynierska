import torch
import numpy as np

def grad(
        image: torch.Tensor,
        model: torch.nn.Module,
        target_class: int = None,
        normalize: bool = True
) -> np.ndarray:
    model.eval()
    image_cp = image.clone().detach()
    image_cp.requires_grad_()
    output = model(image_cp)
    y_c = output[0, target_class]
    grad = torch.autograd.grad(y_c, image_cp)[0]
    result = grad.squeeze().cpu().detach().numpy()
    result = np.maximum(result, 0)
    if normalize:
        result -= np.min(result)
        result /= np.max(result)
    result *= 255.0
    result = result.astype(np.uint8)
    return result