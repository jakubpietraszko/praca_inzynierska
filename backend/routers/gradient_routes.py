import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import APIRouter, Depends
from services.container import get_model_service, get_dataset_service, get_container
from services.model_service import ModelService
from services.dataset_service import DatasetService
from services.container import ServiceContainer
from interfaces.requests.grad import GradRequest
from interfaces.activations import ActivationsRequest, ActivationsResponse

router = APIRouter()


@router.post("/heat-map/grad")
async def grad(
    request: GradRequest,
    model_service: ModelService = Depends(get_model_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
    container: ServiceContainer = Depends(get_container)
):
    try:
        model_info = model_service.load_model(request.model_name)
        model = model_info.model
        transform = model_info.transform
        
        image_path = os.path.join(
            dataset_service.datasets_path,
            request.dataset_name,
            request.label_name,
            request.filename
        )
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Path to image: {image_path}")
        
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        image_cp = image.clone().detach()
        image_cp.requires_grad_()
        
        output = model(image_cp)
        y_c = output[0, request.target_label]
        grad = torch.autograd.grad(y_c, image_cp)[0]
        result = grad.squeeze().cpu().detach().numpy()
        result = np.maximum(result, 0)

        result *= 255.0
        result = result.astype(np.uint8)

        save_path = os.path.join(container.models_datasets_path, 'grad.png')
        plt.imsave(save_path, result, cmap='hot')
        
        return {"success": True, "path": save_path}
        
    except Exception as e:
        print(f"Error generating gradient heatmap: {e}")
        return {"success": False, "error": str(e)}


@router.post("/activation-maps")
async def activation_maps(
    request: ActivationsRequest,
    model_service: ModelService = Depends(get_model_service),
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        model_info = model_service.load_model(request.model)
        model = model_info.model
        transform = model_info.transform
        last_conv = model_info.last_conv

        for image_req in request.images:
            label = image_req.label
            filename = image_req.filename

            path = os.path.join(dataset_service.datasets_path, request.dataset, label, filename)
            base = filename.split('.')[0]
            print(f"Path to image: {path}")
            
            if not os.path.exists(path):
                print(f"Image not found: {path}")
                continue
            
            image = Image.open(path).convert("RGB")
            image = transform(image).unsqueeze(0)

            def activation_hook(module, input, output):
                activation = output.detach().cpu().numpy()
                print(f"Activation shape: {activation.shape}")

            last_conv.register_forward_hook(activation_hook)

            model.eval()
            with torch.no_grad():
                output = model(image)
                output = output.squeeze()
                print(f"Model output shape: {output.shape}")


            

        return ActivationsResponse(success=True)
        
    except Exception as e:
        print(f"Error generating activation maps: {e}")
        return ActivationsResponse(success=False)


@router.get("/heat-map/integrated-grad")
def integrated_grad():
    raise NotImplementedError("Integrated gradients not implemented yet")


@router.get("/heat-map/smooth-grad")
def smooth_grad():
    raise NotImplementedError("Smooth gradients not implemented yet")


@router.get("/heat-map/var-grad")
def var_grad():
    raise NotImplementedError("Variance gradients not implemented yet")


@router.get("/heat-map/grad-cam")
def grad_cam():
    raise NotImplementedError("Grad-CAM not implemented yet")


@router.get("/heat-map/grad-cam-plus-plus")
def grad_cam_plus_plus():
    raise NotImplementedError("Grad-CAM++ not implemented yet")


@router.get("/heat-map/wgrad-cam")
def wgrad_cam():
    raise NotImplementedError("Weighted Grad-CAM not implemented yet")


@router.get("/heat-map/hires-cam")
def hires_cam():
    raise NotImplementedError("HiRes-CAM not implemented yet")


@router.get("/heat-map/ablation-cam")
def ablation_cam():
    raise NotImplementedError("Ablation-CAM not implemented yet")


@router.get("/heat-map/group-cam")
def group_cam():
    raise NotImplementedError("Group-CAM not implemented yet")


@router.get("/heat-map/rise")
def rise():
    raise NotImplementedError("RISE not implemented yet")


@router.get("/heat-map/exact-shap")
def exact_shap():
    raise NotImplementedError("Exact SHAP not implemented yet")


@router.get("/heat-map/kernel-shap")
def kernel_shap():
    raise NotImplementedError("Kernel SHAP not implemented yet")


@router.get("heat-map/lime")
def lime():
    raise NotImplementedError("LIME not implemented yet")
