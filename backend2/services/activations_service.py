import numpy as np
import cv2
import os
from interfaces.activations import ActivationsRequest, ActivationsResponse

class ActivationsService:
    def __init__(self):
        pass

    def activations(self, request: ActivationsRequest,
                    container,
                    models_service,
                    utilities_service,
                    datasets_service,
                    constants
                    ):
        
        model_info = models_service.load_model(request.model)

        filepath = os.path.join(
            container.datasets_path,
            request.dataset,
            request.label,
            request.filename
        )

        image_np = datasets_service.load_image(filepath, model_info.input_shape)
        image_tensor = utilities_service.convert_image_to_tensor(image_np)

        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output.detach().cpu().numpy()

        ref = model_info.last_conv_layer.register_forward_hook(hook)
        model_info.model.eval()
        model_info.model(image_tensor)
        ref.remove()

        activations = np.squeeze(activations, axis=0)

        path = os.path.join(
            container.models_datasets_path,
            utilities_service.models_datasets_path(request.model, request.dataset),
            constants.ACTIVATIONS,
            request.label,
            utilities_service.delete_extension(request.filename)
        )
        os.makedirs(path, exist_ok=True)

        for i, activation in enumerate(activations):
            activation = utilities_service.relu_normalize(
                activation, model_info.input_shape, epsilon=constants.EPSILON)

            activation_path = os.path.join(
                path,
                f'{i}.png'
            )
            cv2.imwrite(activation_path, activation)

        return ActivationsResponse(
            success=True,
            count=len(activations)
        )