import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from sklearn.decomposition import PCA
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/grad-cam', response_model=GradientResponse)
async def grad_cam(request: GradientRequest,
                   container: ServiceContainer = Depends(get_container),
                   models_service: ModelsService = Depends(get_models_service),
                   heatmaps_service: HeatmapsService = Depends(
                       get_heatmaps_service),
                   utilities_service: UtilitiesService = Depends(
                       get_utilities_service),
                   models_datasets_service: ModelsDatasetsService = Depends(
                       get_models_datasets_service),
                   datasets_service: DatasetsService = Depends(
                       get_datasets_service),
                   constants: Constants = Depends(get_constants)
                   ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.GRAD_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=True)
        

        model_info.model.eval()


        activations = None
        gradients = None

        def forward_hook(module, i, o):
            nonlocal activations
            activations = o.clone().detach()

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0].clone().detach()

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)
        model_info.model.eval()
        image_tensor.requires_grad_(True)
        output = model_info.model(image_tensor)
        output[0, target].backward()
        h1.remove()
        h2.remove()



        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = utilities_service.relu_normalize(
            cam, model_info.input_shape, constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, cam)

    return GradientResponse(
        success=True,
    )


@router.post('/grad-cam-plus-plus', response_model=GradientResponse)
async def grad_cam_plus(request: GradientRequest,
                        container: ServiceContainer = Depends(get_container),
                        models_service: ModelsService = Depends(
                            get_models_service),
                        heatmaps_service: HeatmapsService = Depends(
                            get_heatmaps_service),
                        utilities_service: UtilitiesService = Depends(
                            get_utilities_service),
                        models_datasets_service: ModelsDatasetsService = Depends(
                            get_models_datasets_service),
                        datasets_service: DatasetsService = Depends(
                            get_datasets_service),
                        constants: Constants = Depends(get_constants)
                        ):
    pass


@router.post('/group-cam', response_model=GradientResponse)
async def group_cam(request: GradientRequest):
    pass


@router.post('/hires-cam', response_model=GradientResponse)
async def hires_cam(request: GradientRequest,
                    container: ServiceContainer = Depends(get_container),
                    models_service: ModelsService = Depends(
                        get_models_service),
                    heatmaps_service: HeatmapsService = Depends(
                        get_heatmaps_service),
                    utilities_service: UtilitiesService = Depends(
                        get_utilities_service),
                    models_datasets_service: ModelsDatasetsService = Depends(
                        get_models_datasets_service),
                    datasets_service: DatasetsService = Depends(
                        get_datasets_service),
                    constants: Constants = Depends(get_constants)
                    ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.HIRES_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=True)

        model_info.model.eval()

        activations = None
        gradients = None

        def forward_hook(module, i, o):
            nonlocal activations
            activations = o.clone().detach()

        def backward_hook(module, i, o):
            nonlocal gradients
            gradients = o[0].clone().detach()

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        h2 = model_info.last_conv_layer.register_full_backward_hook(backward_hook)

        output = model_info.model(image_tensor)
        output[0, target].backward()
        h1.remove()
        h2.remove()

        weights = gradients

        cam = (activations * weights).sum(dim=1, keepdim=True)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = utilities_service.relu_normalize(
            cam, model_info.input_shape, constants.EPSILON
        )

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename) 
        )
        cv2.imwrite(save_path, cam)
    return GradientResponse(
        success=True,
    )


@router.post('/wgrad-cam', response_model=GradientResponse)
async def wgrad_cam(request: GradientRequest):
    pass


@router.post('/ablation-cam', response_model=GradientResponse)
async def ablation_cam(request: GradientRequest,
                       container: ServiceContainer = Depends(get_container),
                       models_service: ModelsService = Depends(get_models_service),
                       heatmaps_service: HeatmapsService = Depends(
                           get_heatmaps_service),
                       utilities_service: UtilitiesService = Depends(
                           get_utilities_service),
                       models_datasets_service: ModelsDatasetsService = Depends(
                           get_models_datasets_service),
                       datasets_service: DatasetsService = Depends(
                           get_datasets_service),
                       constants: Constants = Depends(get_constants)
                       ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.ABLATION_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=False)

        model_info.model.eval()

        activations = None

        def forward_hook(module, inp, output):
            nonlocal activations
            activations = output

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        
        pred = model_info.model(image_tensor)
        y_c = pred[0, target].item()
        
        h1.remove()

        activations = activations.cpu().detach().numpy()
        weights = np.zeros(activations.shape[1], dtype=np.float32)

        for i in range(activations.shape[1]):
            def hook_zeros_k_map(module, inp, out):
                modified_output = out.clone()
                modified_output[:, i] = 0
                return modified_output

            hook = model_info.last_conv_layer.register_forward_hook(hook_zeros_k_map)
            
            pred_mod = model_info.model(image_tensor)
            y_k = pred_mod[0, target].item()
            
            weights[i] = abs(y_c - y_k) / y_c if y_c != 0 else 0
            
            hook.remove()

        activations = activations[0]
        ablationcam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            ablationcam += float(w) * activations[i, :, :]

        ablationcam = np.maximum(ablationcam, 0)
        ablationcam = cv2.normalize(ablationcam, None, 0, 255, cv2.NORM_MINMAX)

        ablationcam = cv2.resize(ablationcam, (model_info.input_shape[2], model_info.input_shape[1]))

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, ablationcam)

    return GradientResponse(
        success=True,
    )


@router.post('/eigen-cam', response_model=GradientResponse)
async def eigen_cam(request: GradientRequest,
                    container: ServiceContainer = Depends(get_container),
                    models_service: ModelsService = Depends(get_models_service),
                    heatmaps_service: HeatmapsService = Depends(
                        get_heatmaps_service),
                    utilities_service: UtilitiesService = Depends(
                        get_utilities_service),
                    models_datasets_service: ModelsDatasetsService = Depends(
                        get_models_datasets_service),
                    datasets_service: DatasetsService = Depends(
                        get_datasets_service),
                    constants: Constants = Depends(get_constants)
                    ):
    model_info = models_service.load_model(request.model)
    path = os.path.join(
        container.models_datasets_path,
        utilities_service.models_datasets_path(request.model, request.dataset),
        constants.EIGEN_CAM,
    )
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label = image.label
        filename = image.filename
        target = image.target

        label_path = os.path.join(
            path,
            label,
        )
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(
            container.datasets_path,
            request.dataset,
            label,
            filename
        )

        image_np = datasets_service.load_image(
            image_path, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(
            image_np, grad=False)

        model_info.model.eval()

        # Extract feature maps from the last convolutional layer
        activations = None

        def forward_hook(module, inp, output):
            nonlocal activations
            activations = output

        h1 = model_info.last_conv_layer.register_forward_hook(forward_hook)
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = model_info.model(image_tensor)
        
        h1.remove()

        # Convert to numpy and get feature maps
        feature_maps = activations.cpu().detach().numpy()[0]  # Remove batch dimension
        
        # Reshape feature maps into 2D matrix (channels × spatial locations)
        num_channels, height, width = feature_maps.shape
        feature_matrix = feature_maps.reshape(num_channels, height * width)
        
        # Transpose to have spatial locations as rows and channels as columns for PCA
        feature_matrix = feature_matrix.T
        
        # Apply PCA to find principal components
        pca = PCA(n_components=1)
        pca.fit(feature_matrix)
        
        # Get the first principal component (eigenvector)
        first_component = pca.components_[0]  # Shape: (num_channels,)
        
        # Project feature maps onto the first principal component
        eigen_cam = np.zeros((height, width), dtype=np.float32)
        
        for i in range(num_channels):
            eigen_cam += first_component[i] * feature_maps[i]
        
        # Apply ReLU to keep only positive contributions
        eigen_cam = np.maximum(eigen_cam, 0)
        
        # Normalize the heatmap
        if eigen_cam.max() > 0:
            eigen_cam = (eigen_cam - eigen_cam.min()) / (eigen_cam.max() - eigen_cam.min())
        
        # Convert to 0-255 range
        eigen_cam = (eigen_cam * 255).astype(np.uint8)
        
        # Upsample to the size of the input image
        eigen_cam = cv2.resize(eigen_cam, (model_info.input_shape[2], model_info.input_shape[1]))

        save_path = os.path.join(
            label_path,
            utilities_service.add_png_extension(filename)
        )
        cv2.imwrite(save_path, eigen_cam)

    return GradientResponse(
        success=True,
    )
