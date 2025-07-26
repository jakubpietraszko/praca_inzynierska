import torch
import numpy as np

def smoothgrad(
        image: torch.Tensor,
        model: torch.nn.Module,
        target_class: int = None,
        normalize: bool = True,
        num_samples: int = 1000,
        noise_level: float = 0.01
) -> np.ndarray:
    result = torch.zeros_like(image)
    model.eval()
    for _ in range(num_samples):
        noise = torch.randn_like(image) * noise_level
        noisy_image = (image + noise).clone().detach()
        noisy_image.requires_grad_()
        output = model(noisy_image)
        y_c = output[0, target_class]
        grad = torch.autograd.grad(y_c, noisy_image)[0]
        result += grad
    result /= num_samples
    result = result.squeeze().cpu().detach().numpy()
    result = np.maximum(result, 0)
    if normalize:
        result -= np.min(result)
        result /= np.max(result)
    result *= 255.0
    result = result.astype(np.uint8)
    return result

