import torch
import numpy as np

def ablationcam(
        image: torch.Tensor,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        target_class: int = None,
        normalize: bool = True,
        background: int = 0
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
    # print(output.shape)

    h1.remove()
    h2.remove()
    softmax = torch.softmax(
        output, dim=1
    )

    y_c = softmax[0, target_class].item()

    numberOfFeaturesMaps = activations.shape[1]

    weights = torch.ones_like(activations)
    # print(weights.shape)

    for i in range(numberOfFeaturesMaps):
        def hookKMap(module, inp, out):
            modified_output = out.clone()
            modified_output[:, i] = background
            return modified_output

        h = target_layer.register_forward_hook(hookKMap)

        pred_mod = model(image)
        softmax = torch.softmax(pred_mod, dim=1)
        y_k = softmax[0, target_class]
        w = (y_c - y_k if y_c > y_k else y_k - y_c) / (y_c + 1e-3)
        # print(y_c, y_k.item(), w.item())
        weights[0, i, :, :] *= w

        h.remove()

    cam = (activations * weights).sum(dim=1, keepdim=True)
    cam = cam.squeeze().cpu().detach().numpy()
    # cam = np.maximum(cam, 0)

    # if normalize:
    #     cam -= np.min(cam)
    #     cam /= np.max(cam)

    # cam *= 255.0
    # cam = cam.astype(np.uint8)
    
    return cam
    









