import builtins
import contextlib
import os
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge

from interfaces.gradient import GradientRequest, GradientResponse
from interfaces.label_filename import LabelFilename
from interfaces.super_pixels import SuperPixelsRequest
from services.config_service import XAIMethodConfig
from services.models_service import ModelInfo


class ProbingImageService:
    def __init__(self):
        from services.container import (
            ConfigService,
            Constants,
            DatasetsService,
            LoggerService,
            MasksService,
            ModelsService,
            ServiceContainer,
            UtilitiesService,
            WebSocketService,
            get_config_service,
            get_constants,
            get_container,
            get_datasets_service,
            get_logger_service,
            get_masks_service,
            get_models_service,
            get_utilities_service,
            get_websocket_service,
        )

        self.config_service: ConfigService = get_config_service()
        self.get_masks_service: MasksService = get_masks_service()
        self.websocket_service: WebSocketService = get_websocket_service()
        self.constants: Constants = get_constants()
        self.container: ServiceContainer = get_container()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.models_service: ModelsService = get_models_service()
        self.logger_service: LoggerService = get_logger_service()
        self.utilities_service: UtilitiesService = get_utilities_service()

    async def __make_root_path(self, request: GradientRequest, xai_method_name: str) -> str:
        root_path: str = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            xai_method_name,
        )
        os.makedirs(root_path, exist_ok=True)
        return root_path

    async def __make_label_path(self, root_path: str, label: str) -> str:
        label_path: str = os.path.join(root_path, label)
        os.makedirs(label_path, exist_ok=True)
        return label_path

    async def __make_superpixels(
        self, request: GradientRequest, label: str, filename: str, config: XAIMethodConfig
    ) -> np.ndarray | None:
        request_superpixels: SuperPixelsRequest = SuperPixelsRequest(
            model=request.model,
            dataset=request.dataset,
            images=[LabelFilename(label=label, filename=filename)],
            superPixels=(config.super_pixels or self.constants.DEFAULT_SUPER_PIXELS),
            overwrite=False,
        )

        self.logger_service.log_config(config, f"Superpixels for {label}/{filename}")

        if config.segmentation is None:
            return None
        elif config.segmentation == self.constants.MASK_SEED:
            await self.get_masks_service.seed_with_save(request=request_superpixels)
        elif config.segmentation == self.constants.MASK_SLIC:
            await self.get_masks_service.slic_with_save(request=request_superpixels)
        else:
            return None

        super_pixels_path = os.path.join(
            self.container.models_datasets_path,
            await self.utilities_service.models_datasets_path(request.model, request.dataset),
            config.segmentation,
            label,
            filename,
        )

        super_pixels_np = np.array(Image.open(super_pixels_path))

        self.logger_service.log(f"{super_pixels_np.shape}")

        unique_values = np.unique(super_pixels_np)
        super_pixels_segments = np.zeros_like(super_pixels_np)
        for i, val in enumerate(unique_values):
            super_pixels_segments[super_pixels_np == val] = i
        return super_pixels_segments

    async def xrai(self, request: GradientRequest) -> GradientResponse:
        root_path = await self.__make_root_path(request, self.constants.XRAI)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.XRAI)

        self.logger_service.log_config(config, self.constants.XRAI)

        for image in request.images:
            label, filename, target = image.label, image.filename, image.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = await self.__make_label_path(root_path, label)

            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))
            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            super_pixels_segments = await self.__make_superpixels(request, label, filename, config)

            if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3]:
                image_np_hwc = np.transpose(image_np, (1, 2, 0))
            else:
                image_np_hwc = image_np

            image_tensor = await self.utilities_service.convert_image_to_tensor(image_np, grad=True)
            model_info.model.eval()
            output = model_info.model(image_tensor)
            y_c = output[0, target]
            grad = torch.autograd.grad(y_c, image_tensor)[0]
            attribution_map = grad.squeeze().cpu().detach().numpy()

            if len(attribution_map.shape) == 3 and attribution_map.shape[0] == 3:
                attribution_map = await self.utilities_service.aggregate_channels(
                    attribution_map, config.aggregation or self.constants.DEFAULT_AGGREGATION
                )

            attribution_map = np.abs(attribution_map)

            if attribution_map.shape != super_pixels_segments.shape:
                attribution_map = cv2.resize(
                    attribution_map, (super_pixels_segments.shape[1], super_pixels_segments.shape[0])
                )

            xrai_result = await self._xrai_algorithm(
                image=image_np_hwc, segments=super_pixels_segments, attribution_map=attribution_map
            )

            if np.max(xrai_result) > np.min(xrai_result):
                xrai_normalized = (xrai_result - np.min(xrai_result)) / (np.max(xrai_result) - np.min(xrai_result))
                xrai_normalized = xrai_normalized * 255.0
                xrai_final = xrai_normalized.astype(np.uint8)

                if xrai_final.shape != (model_info.input_shape[1], model_info.input_shape[2]):
                    xrai_final = cv2.resize(xrai_final, (model_info.input_shape[2], model_info.input_shape[1]))

            else:
                xrai_final = np.zeros((model_info.input_shape[1], model_info.input_shape[2]), dtype=np.uint8)

            xrai_final = xrai_final.astype(np.uint8)

            await self.utilities_service.save_image(save_path, xrai_final)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request,
                label=label,
                filename=filename,
                heatmap_type=self.constants.XRAI,
            )

        return GradientResponse(success=True)

    async def kernel_shap(self, request: GradientRequest) -> GradientResponse:
        root_path: str = await self.__make_root_path(request, self.constants.KERNEL_SHAP)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.KERNEL_SHAP)

        self.logger_service.log_config(config, self.constants.KERNEL_SHAP)

        for image in request.images:
            label, filename, target = image.label, image.filename, image.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = await self.__make_label_path(root_path, label)
            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            super_pixels_segments = await self.__make_superpixels(request, label, filename, config)

            if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3]:
                image_np_hwc = np.transpose(image_np, (1, 2, 0))
            else:
                image_np_hwc = image_np

            shap_values = await self._kernel_shap_explain(
                image_np=image_np_hwc,
                super_pixels=super_pixels_segments,
                model=model_info.model,
                target=target,
                utilities_service=self.utilities_service,
                num_samples=(config.n_samples or self.constants.DEFAULT_N_SAMPLES),
            )

            if np.max(shap_values) > np.min(shap_values):
                shap_normalized = (shap_values - np.min(shap_values)) / (np.max(shap_values) - np.min(shap_values))
                shap_normalized = shap_normalized * 255.0
                shap_final = shap_normalized.astype(np.uint8)

                if shap_final.shape != (model_info.input_shape[1], model_info.input_shape[2]):
                    shap_final = cv2.resize(shap_final, (model_info.input_shape[2], model_info.input_shape[1]))

            else:
                shap_final = np.zeros((model_info.input_shape[1], model_info.input_shape[2]), dtype=np.uint8)

            shap_final = shap_final.astype(np.uint8)

            await self.utilities_service.save_image(save_path, shap_final)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request, label=label, filename=filename, heatmap_type=self.constants.KERNEL_SHAP
            )

        return GradientResponse(success=True)

    async def lime(self, request: GradientRequest) -> GradientResponse:
        root_path: str = await self.__make_root_path(request, self.constants.LIME)

        model_info: ModelInfo = await self.models_service.load_model(request.model)

        config: XAIMethodConfig = await self.config_service.get_method_config(self.constants.LIME)

        self.logger_service.log_config(config, self.constants.LIME)

        for image in request.images:
            label, filename, target = image.label, image.filename, image.target

            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            label_path = await self.__make_label_path(root_path, label)

            save_path = os.path.join(label_path, await self.utilities_service.add_png_extension(filename))

            if self.constants.CACHE_ENABLED and await self.utilities_service.skip_forward(save_path):
                continue

            image_path = os.path.join(self.container.datasets_path, request.dataset, label, filename)

            image_np = await self.datasets_service.load_image(image_path, model_info.input_shape)

            super_pixels_segments = await self.__make_superpixels(request, label, filename, config)

            if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3]:
                image_np_hwc = np.transpose(image_np, (1, 2, 0))
            else:
                image_np_hwc = image_np

            attribution = await self._lime_explain(
                image_np=image_np_hwc,
                super_pixels=super_pixels_segments,
                model=model_info.model,
                target=target,
                utilities_service=self.utilities_service,
                num_samples=(config.n_samples or self.constants.DEFAULT_N_SAMPLES),
            )

            if np.max(attribution) > np.min(attribution):
                attribution_normalized = (attribution - np.min(attribution)) / (
                    np.max(attribution) - np.min(attribution)
                )
                attribution_normalized = attribution_normalized * 255.0
                attribution_final = attribution_normalized.astype(np.uint8)

                if attribution_final.shape != (model_info.input_shape[1], model_info.input_shape[2]):
                    attribution_final = cv2.resize(
                        attribution_final, (model_info.input_shape[2], model_info.input_shape[1])
                    )
            else:
                attribution_final = np.zeros((model_info.input_shape[1], model_info.input_shape[2]), dtype=np.uint8)

            attribution_final = attribution_final.astype(np.uint8)

            await self.utilities_service.save_image(save_path, attribution_final)

            await self.websocket_service.send_notification_supress_exceptions(
                request=request, label=label, filename=filename, heatmap_type=self.constants.LIME
            )

        return GradientResponse(success=True)

    async def _lime_explain(self, image_np, super_pixels, model, target, utilities_service, num_samples=1000):
        unique_segments = np.unique(super_pixels)
        n_segments = len(unique_segments)

        if n_segments < 2:
            return np.zeros((super_pixels.shape[0], super_pixels.shape[1]), dtype=np.float32)

        if len(image_np.shape) == 3 and image_np.shape[2] in [1, 3]:
            image_model_format = np.transpose(image_np, (2, 0, 1))
        else:
            image_model_format = np.expand_dims(image_np, axis=0) if len(image_np.shape) == 2 else image_np

        image_tensor = await utilities_service.convert_image_to_tensor(image_model_format)
        model.eval()
        with torch.no_grad():
            original_pred = model(image_tensor)[0, target].item()

        perturbations = []
        predictions = []

        perturbations.append(np.ones(n_segments))
        predictions.append(original_pred)

        empty_image = await self._apply_perturbation(
            image_np, super_pixels, unique_segments, np.zeros(n_segments).astype(np.uint8)
        )
        empty_tensor = await utilities_service.convert_image_to_tensor(empty_image)
        with torch.no_grad():
            empty_pred = model(empty_tensor)[0, target].item()
        perturbations.append(np.zeros(n_segments))
        predictions.append(empty_pred)

        for i in range(min(num_samples - 2, 100)):
            prob = np.random.uniform(0.3, 0.7)
            mask = np.random.binomial(1, prob, n_segments).astype(np.uint8)

            perturbed_image = await self._apply_perturbation(image_np, super_pixels, unique_segments, mask)

            perturbed_tensor = await utilities_service.convert_image_to_tensor(perturbed_image)
            with torch.no_grad():
                pred = model(perturbed_tensor)[0, target].item()

            perturbations.append(mask)
            predictions.append(pred)

        perturbations = np.array(perturbations)
        predictions = np.array(predictions)

        ridge = Ridge(alpha=0.1, fit_intercept=True)
        ridge.fit(perturbations, predictions)

        coefficients = ridge.coef_

        explanation = np.zeros_like(super_pixels, dtype=np.float32)

        for i, segment_id in enumerate(unique_segments):
            mask = super_pixels == segment_id
            explanation[mask] = coefficients[i]

        explanation = np.abs(explanation)

        return explanation

    async def _apply_perturbation(
        self, image: np.ndarray, super_pixels: np.ndarray, unique_segments: np.ndarray, mask: np.ndarray
    ):
        assert len(image.shape) == 3 and image.shape[2] in [1, 3]
        assert type(image) is np.ndarray and image.dtype == np.uint8

        assert super_pixels.dtype == np.uint8
        assert unique_segments.dtype == np.uint8  # 1d vector [0, 1, 2, .... n]
        assert mask.dtype == np.uint8  # 1d vector of 0,1

        perturbed_image = image.copy().astype(np.float32)
        BASELINE = 0

        segment_mask_values = mask[super_pixels]

        perturbation_mask = segment_mask_values == 0

        if len(image.shape) == 3:
            if image.shape[2] == 3 or image.shape[2] == 1:
                perturbed_image = np.where(perturbation_mask[..., np.newaxis], BASELINE, perturbed_image)
        else:
            perturbed_image = np.where(perturbation_mask, BASELINE, perturbed_image)

        result = perturbed_image.astype(np.uint8)

        if len(result.shape) == 3 and result.shape[2] in [1, 3]:
            result = np.transpose(result, (2, 0, 1))
        elif len(result.shape) == 2:
            result = np.expand_dims(result, axis=0)

        return result

    async def _xrai_algorithm(self, image, segments, attribution_map):
        H, W = segments.shape
        unique_segments = np.unique(segments)
        M = np.zeros_like(segments, dtype=bool)
        S = set(unique_segments)
        T = []
        segment_masks = {}
        segment_areas = {}
        segment_attributions = {}

        for seg_id in unique_segments:
            mask = segments == seg_id
            segment_masks[seg_id] = mask
            segment_areas[seg_id] = np.sum(mask)
            segment_attributions[seg_id] = np.sum(attribution_map[mask])

        iteration = 0
        max_iterations = len(unique_segments)

        final_attribution = np.zeros_like(segments, dtype=np.float32)

        while S and np.sum(M) < H * W and iteration < max_iterations:
            gains = {}

            for s in S:
                segment_mask = segment_masks[s]
                remaining_mask = segment_mask & (~M)

                if np.sum(remaining_mask) > 0:
                    attribution_sum = np.sum(attribution_map[remaining_mask])
                    area = np.sum(remaining_mask)
                    gains[s] = attribution_sum / area if area > 0 else 0
                else:
                    gains[s] = 0

            if not gains or all(g <= 0 for g in gains.values()):
                if gains and any(g >= 0 for g in gains.values()):
                    s_hat = max(gains, key=gains.get)
                else:
                    break
            else:
                s_hat = max(gains, key=gains.get)

            S.remove(s_hat)
            new_region = segment_masks[s_hat] & (~M)
            M = M | segment_masks[s_hat]

            importance_score = len(unique_segments) - iteration
            final_attribution[new_region] = importance_score

            T.append(M.copy())

            iteration += 1

        if np.max(final_attribution) == 0:
            for seg_id in unique_segments:
                mask = segment_masks[seg_id]
                if np.sum(mask) > 0:
                    avg_attribution = np.mean(attribution_map[mask])
                    final_attribution[mask] = avg_attribution

        return final_attribution

    async def _kernel_shap_explain(self, image_np, super_pixels, model, target, utilities_service, num_samples=1000):
        unique_segments = np.unique(super_pixels)
        n_segments = len(unique_segments)

        if n_segments < 2:
            return np.zeros((super_pixels.shape[0], super_pixels.shape[1]), dtype=np.float32)

        if len(image_np.shape) == 3 and image_np.shape[2] in [1, 3]:
            image_model_format = np.transpose(image_np, (2, 0, 1))
        else:
            image_model_format = np.expand_dims(image_np, axis=0) if len(image_np.shape) == 2 else image_np

        empty_image = await self._apply_perturbation(
            image_np, super_pixels, unique_segments, np.zeros(n_segments).astype(np.uint8)
        )
        empty_tensor = await utilities_service.convert_image_to_tensor(empty_image)
        model.eval()
        with torch.no_grad():
            baseline_pred = model(empty_tensor)[0, target].item()

        image_tensor = await utilities_service.convert_image_to_tensor(image_model_format)
        with torch.no_grad():
            full_pred = model(image_tensor)[0, target].item()

        coalitions = []
        predictions = []
        weights = []

        coalitions.append(np.zeros(n_segments))
        predictions.append(baseline_pred)
        weights.append(await self._shapley_kernel_weight(0, n_segments))

        coalitions.append(np.ones(n_segments))
        predictions.append(full_pred)
        weights.append(await self._shapley_kernel_weight(n_segments, n_segments))

        for _ in range(min(num_samples - 2, 2000)):
            coalition_size = np.random.choice(
                range(1, n_segments), p=await self._coalition_sampling_distribution(n_segments)
            )

            coalition = np.zeros(n_segments)
            selected_segments = np.random.choice(n_segments, coalition_size, replace=False)
            coalition[selected_segments] = 1

            perturbed_image = await self._apply_perturbation(
                image_np, super_pixels, unique_segments, coalition.astype(np.uint8)
            )
            perturbed_tensor = await utilities_service.convert_image_to_tensor(perturbed_image)

            with torch.no_grad():
                pred = model(perturbed_tensor)[0, target].item()

            coalitions.append(coalition)
            predictions.append(pred)
            weights.append(await self._shapley_kernel_weight(coalition_size, n_segments))

        coalitions = np.array(coalitions)
        predictions = np.array(predictions)
        weights = np.array(weights)

        try:
            from sklearn.linear_model import Ridge

            ridge = Ridge(alpha=0.01, fit_intercept=True)

            sample_weight = weights / np.sum(weights) * len(weights)
            ridge.fit(coalitions, predictions, sample_weight=sample_weight)

            coefficients = ridge.coef_

        except Exception:
            ridge = Ridge(alpha=0.1, fit_intercept=True)
            ridge.fit(coalitions, predictions)
            coefficients = ridge.coef_

        explanation = np.zeros_like(super_pixels, dtype=np.float32)
        for i, segment_id in enumerate(unique_segments):
            mask = super_pixels == segment_id
            explanation[mask] = abs(coefficients[i])

        return explanation

    async def _shapley_kernel_weight(self, coalition_size, n_segments):
        if coalition_size == 0 or coalition_size == n_segments:
            return 1000.0

        weight = (n_segments - 1) / (coalition_size * (n_segments - coalition_size))
        return weight

    async def _coalition_sampling_distribution(self, n_segments):
        sizes = np.arange(1, n_segments)
        weights = np.array([1.0 / (size * (n_segments - size)) for size in sizes])
        weights = weights / np.sum(weights)
        return weights
