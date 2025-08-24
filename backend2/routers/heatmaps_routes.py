import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()



# @router.post('/lrp-epsilon', response_model=GradientResponse)
# async def lrp_epsilon(request: GradientRequest,
#                       container: ServiceContainer = Depends(get_container),
#                       models_service: ModelsService = Depends(get_models_service),
#                       heatmaps_service: HeatmapsService = Depends(
#                           get_heatmaps_service),
#                       utilities_service: UtilitiesService = Depends(
#                           get_utilities_service),
#                       models_datasets_service: ModelsDatasetsService = Depends(
#                           get_models_datasets_service),
#                       datasets_service: DatasetsService = Depends(
#                           get_datasets_service),
#                       constants: Constants = Depends(get_constants)
#                       ):
#     model_info = models_service.load_model(request.model)

#     path = os.path.join(
#         container.models_datasets_path,
#         utilities_service.models_datasets_path(request.model, request.dataset),
#         constants.EPSILON_LRP,
#     )
#     os.makedirs(path, exist_ok=True)

#     for image in request.images:
#         label = image.label
#         filename = image.filename
#         target = image.target

#         label_path = os.path.join(
#             path,
#             label,
#         )
#         os.makedirs(label_path, exist_ok=True)

#         image_path = os.path.join(
#             container.datasets_path,
#             request.dataset,
#             label,
#             filename
#         )

#         image_np = datasets_service.load_image(
#             image_path, model_info.input_shape)
#         image_tensor = utilities_service.convert_image_to_tensor(
#             image_np, grad=True)

#         model_info.model.eval()

#         def lrp_epsilon_layer(layer, input_activation, output_relevance, epsilon=1e-8):
            
#             if isinstance(layer, torch.nn.Linear):
#                 weight = layer.weight.data
#                 bias = layer.bias.data if layer.bias is not None else None
                
#                 z = torch.mm(input_activation, weight.t())
#                 if bias is not None:
#                     z = z + bias
                
#                 z_eps = z + epsilon * torch.sign(z)
                
#                 s = output_relevance / z_eps
#                 c = torch.mm(s, weight)
#                 input_relevance = input_activation * c
                
#                 return input_relevance
                
#             elif isinstance(layer, torch.nn.Conv2d):
#                 weight = layer.weight.data
#                 bias = layer.bias.data if layer.bias is not None else None
                
#                 z = torch.nn.functional.conv2d(
#                     input_activation, weight, bias,
#                     stride=layer.stride, padding=layer.padding,
#                     dilation=layer.dilation, groups=layer.groups
#                 )
                
#                 z_eps = z + epsilon * torch.sign(z)
                
#                 s = output_relevance / z_eps
#                 c = torch.nn.functional.conv_transpose2d(
#                     s, weight, None,
#                     stride=layer.stride, padding=layer.padding,
#                     output_padding=0, groups=layer.groups, dilation=layer.dilation
#                 )
                
#                 if c.shape != input_activation.shape:
#                     if c.shape[2] > input_activation.shape[2] or c.shape[3] > input_activation.shape[3]:
#                         c = c[:, :, :input_activation.shape[2], :input_activation.shape[3]]
#                     elif c.shape[2] < input_activation.shape[2] or c.shape[3] < input_activation.shape[3]:
#                         pad_h = input_activation.shape[2] - c.shape[2]
#                         pad_w = input_activation.shape[3] - c.shape[3]
#                         c = torch.nn.functional.pad(c, (0, pad_w, 0, pad_h))
                
#                 input_relevance = input_activation * c
#                 return input_relevance
                
#             elif isinstance(layer, torch.nn.ReLU):
#                 return output_relevance
                
#             elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)):
#                 if isinstance(layer, torch.nn.MaxPool2d):
#                     return torch.nn.functional.interpolate(
#                         output_relevance, size=input_activation.shape[2:], mode='nearest'
#                     )
#                 else:
#                     return torch.nn.functional.interpolate(
#                         output_relevance, size=input_activation.shape[2:], mode='bilinear', align_corners=False
#                     )
            
#             elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
#                 return output_relevance
                
#             elif isinstance(layer, torch.nn.Dropout):
#                 return output_relevance
                
#             else:
#                 return output_relevance

#         activations = []
#         layers = []
        
#         def forward_hook(name):
#             def hook(module, input, output):
#                 activations.append((name, input[0].clone().detach() if isinstance(input, tuple) else input.clone().detach()))
#                 layers.append((name, module))
#             return hook
        
#         hooks = []
#         for name, module in model_info.model.named_modules():
#             if len(list(module.children())) == 0:
#                 hook = module.register_forward_hook(forward_hook(name))
#                 hooks.append(hook)
        
#         try:
#             with torch.no_grad():
#                 output = model_info.model(image_tensor)
            
#             for hook in hooks:
#                 hook.remove()
            
#             relevance = torch.zeros_like(output)
#             relevance[0, target] = output[0, target]
            
#             for i in reversed(range(len(layers))):
#                 layer_name, layer = layers[i]
#                 activation_name, activation = activations[i]
                
#                 if i == len(layers) - 1:
#                     if isinstance(layer, torch.nn.Linear):
#                         if i > 0:
#                             prev_activation = activations[i-1][1]
#                             relevance = lrp_epsilon_layer(layer, prev_activation, relevance, constants.EPSILON)
#                 else:
#                     if i > 0:
#                         prev_activation = activations[i-1][1]
#                         relevance = lrp_epsilon_layer(layer, prev_activation, relevance, constants.EPSILON)
#                     else:
#                         relevance = lrp_epsilon_layer(layer, image_tensor, relevance, constants.EPSILON)
            
#             result = relevance.squeeze().cpu().detach().numpy()
            
#         except Exception as e:
#             print(f"LRP failed, falling back to gradient: {e}")
#             output = model_info.model(image_tensor)
#             y_c = output[0, target]
#             grad = torch.autograd.grad(y_c, image_tensor)[0]
#             result = grad.squeeze().cpu().detach().numpy()

#         if len(result.shape) == 3 and result.shape[0] == 3:
#             result = utilities_service.aggregate_channels(result, 'sum')

#         result = utilities_service.relu_normalize(
#             result, model_info.input_shape, epsilon=constants.EPSILON
#         )

#         save_path = os.path.join(
#             label_path,
#             utilities_service.add_png_extension(filename)
#         )

#         cv2.imwrite(save_path, result)

#     return GradientResponse(
#         success=True,
#     )