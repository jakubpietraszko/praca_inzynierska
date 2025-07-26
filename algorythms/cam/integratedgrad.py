import torch
import numpy as np

def integratedgrad(
        image: torch.Tensor,
        model: torch.nn.Module,
        target_class: int = None,
        background: int = 0,
        steps: int = 50,
        normalize: bool = True
) -> np.ndarray:
    model.eval()
    image_cp = image.clone().detach()
    image_cp = image_cp.unsqueeze(0)
    baseline = torch.ones_like(image_cp) * background
    print(image_cp.shape)
    print(baseline.shape)
    alphas = torch.linspace(0, 1, steps).reshape(-1, 1, 1, 1)
    print(alphas.shape)
    batch_size, channels, height, width = image_cp.shape
    interpolated_images = baseline + alphas * (image_cp - baseline)

    print(interpolated_images.shape)

    result = torch.zeros_like(image_cp)

    for i in range(steps):
        interpolated_image = interpolated_images[i].requires_grad_()
        output = model(interpolated_image.unsqueeze(0))
        y_c = output[0, target_class]
        grad = torch.autograd.grad(y_c, interpolated_image)[0]
        result += grad
    result /= steps
    result = result.squeeze().cpu().detach().numpy()
    result = np.maximum(result, 0)
    if normalize:
        result -= np.min(result)
        result /= np.max(result)
    result *= 255.0
    result = result.astype(np.uint8)
    return result
