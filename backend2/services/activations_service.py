import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, normalize

from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.unsupervised_activations import (
    LabelFilenameCoords,
    UnsupervisedActivationsRequest,
    UnsupervisedActivationsResponse,
)


class ActivationsService:
    def __init__(self):
        from constans import Constants
        from services.container import (
            get_constants,
            get_container,
            get_datasets_service,
            get_models_service,
            get_utilities_service,
        )
        from services.datasets_service import DatasetsService
        from services.models_service import ModelsService
        from services.utilities_service import UtilitiesService

        self.utilities_service: UtilitiesService = get_utilities_service()
        self.container = get_container()
        self.constants: Constants = get_constants()
        self.models_service: ModelsService = get_models_service()
        self.datasets_service: DatasetsService = get_datasets_service()

    async def activations(
        self,
        request: ActivationsRequest,
    ):
        model_info = await self.models_service.load_model(request.model)

        filepath = os.path.join(self.container.datasets_path, request.dataset, request.label, request.filename)

        image_np = await self.datasets_service.load_image(filepath, model_info.input_shape)
        image_tensor = await self.utilities_service.convert_image_to_tensor(image_np)

        activations: torch.Tensor | None = None

        def hook(module, input, output):
            nonlocal activations
            activations = output.detach().cpu().numpy()

        ref = model_info.last_conv_layer.register_forward_hook(hook)
        model_info.model.eval()
        model_info.model(image_tensor)
        ref.remove()

        if activations is None:
            raise RuntimeError("Failed to obtain activations from the model.")

        activations = np.squeeze(activations, axis=0)

        path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.ACTIVATIONS,
            request.label,
            await self.utilities_service.delete_extension(request.filename),
        )
        os.makedirs(path, exist_ok=True)

        for i, activation in enumerate(activations):
            activation = await self.utilities_service.relu_normalize(
                activation, model_info.input_shape, epsilon=self.constants.EPSILON
            )

            activation_path = os.path.join(path, f"{i}.png")
            cv2.imwrite(activation_path, activation)

        return ActivationsResponse(success=True, count=len(activations))

    async def unsupervised_activations(self, request: UnsupervisedActivationsRequest):
        req: ActivationsRequest = ActivationsRequest(
            model=request.model, dataset=request.dataset, label=request.label, filename=request.filename
        )

        await self.activations(request=req)

        images_dir = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            self.constants.ACTIVATIONS,
            request.label,
            await self.utilities_service.delete_extension(request.filename),
        )

        image_path = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        images = []

        for img_path in image_path:
            img = np.array(Image.open(img_path))
            images.append(img)

        images_np = np.array(images)
        n_samples = images_np.shape[0]
        images_flat_np = images_np.reshape(n_samples, -1)
        tsne = TSNE(n_components=2)
        images_tsne = tsne.fit_transform(images_flat_np)
        images_tsne = normalize(images_tsne, axis=0)
        scaler = MinMaxScaler()
        images_tsne = scaler.fit_transform(images_tsne)
        label_filenames_coords = []

        for img_path, coords in zip(image_path, images_tsne, strict=False):
            label_filenames_coords.append(
                LabelFilenameCoords(
                    label=request.label,
                    filename=os.path.basename(img_path),
                    coordX=float(coords[0]),
                    coordY=float(coords[1]),
                )
            )

        return UnsupervisedActivationsResponse(
            dataset=request.dataset, label_filenames_coords=label_filenames_coords, success=True
        )
