import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from interfaces.heatmap import (
    HeatmapRequest,
    HeatmapResponse,
    HeatmapStatisticRequest,
    HeatmapStatisticResponse,
    HeatmapStatisticsDatasetManyRequest,
    HeatmapStatisticsDatasetManyResponse,
    HeatmapStatisticsDatasetRequest,
    HeatmapStatisticsDatasetResponse,
    HeatmapStatisticsLabelManyRequest,
    HeatmapStatisticsLabelManyResponse,
    HeatmapStatisticsLabelRequest,
    HeatmapStatisticsLabelResponse,
    LabelFilenameTarget,
)
from interfaces.insertion_deletion import InsertionDeletionRequest, InsertionDeletionResponse, ListOfScores
from interfaces.shapes import (
    ShapeMetricsDatasetRequest,
    ShapeMetricsDatasetResponse,
    ShapeMetricsLabelRequest,
    ShapeMetricsLabelResponse,
)
from services.models_service import ModelInfo


class HeatmapsFacedeService:
    def __init__(self):
        from services.container import (
            Constants,
            DatasetsService,
            ModelsService,
            ServiceContainer,
            UtilitiesService,
            get_cam_service,
            get_constants,
            get_container,
            get_datasets_service,
            get_gradient_noise_service,
            get_gradient_service,
            get_integrated_gradients_service,
            get_lrp_service,
            get_models_service,
            get_probing_image_service,
            get_utilities_service,
        )

        self.get_gradient_service = get_gradient_service
        self.get_integrated_gradients_service = get_integrated_gradients_service
        self.get_gradient_noise_service = get_gradient_noise_service
        self.get_cam_service = get_cam_service
        self.get_lrp_service = get_lrp_service
        self.get_probing_image_service = get_probing_image_service

        self.get_container = get_container
        self.get_datasets_service = get_datasets_service
        self.get_utilities_service = get_utilities_service
        self.get_constants = get_constants
        self.get_models_service = get_models_service
        self.model_service: ModelsService = get_models_service()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.container: ServiceContainer = get_container()
        self.utilities_service: UtilitiesService = get_utilities_service()
        self.constants: Constants = get_constants()

    async def generate_heatmap(
        self,
        request: HeatmapRequest,
        type: str,
    ) -> HeatmapResponse:
        if type == self.get_constants().GRADIENT:
            return await self.get_gradient_service().gradient(request)
        elif type == self.get_constants().GRADIENT_SQUARED:
            return await self.get_gradient_service().gradient_squared(request)
        elif type == self.get_constants().DECONVOLUTION:
            return await self.get_gradient_service().deconvolution(request)
        elif type == self.get_constants().GUIDED_BACKPROPAGATION:
            return await self.get_gradient_service().guided_backpropagation(request)
        elif type == self.get_constants().INPUT_GRADIENT:
            return await self.get_gradient_service().input_gradient(request)

        elif type == self.get_constants().INTEGRATED_GRADIENT:
            return await self.get_integrated_gradients_service().integrated_gradient(request)

        elif type == self.get_constants().SMOOTH_GRADIENT:
            return await self.get_gradient_noise_service().smooth_grad(request)
        elif type == self.get_constants().SMOOTH_GRADIENT_SQUARE:
            return await self.get_gradient_noise_service().smooth_grad_squared(request)
        elif type == self.get_constants().VAR_GRADIENT:
            return await self.get_gradient_noise_service().var_grad(request)

        elif type == self.get_constants().GRAD_CAM:
            return await self.get_cam_service().grad_cam(request)
        elif type == self.get_constants().GRAD_CAM_PLUS_PLUS:
            return await self.get_cam_service().grad_cam_plus_plus(request)
        elif type == self.get_constants().GROUP_CAM:
            return await self.get_cam_service().group_cam(request)
        elif type == self.get_constants().HIRES_CAM:
            return await self.get_cam_service().hires_cam(request)
        elif type == self.get_constants().WGRAD_CAM:
            return await self.get_cam_service().wgrad_cam(request)
        elif type == self.get_constants().ABLATION_CAM:
            return await self.get_cam_service().ablation_cam(request)
        elif type == self.get_constants().XGRAD_CAM:
            return await self.get_cam_service().xgrad_cam(request)

        elif type == self.get_constants().LIME:
            return await self.get_probing_image_service().lime(request)
        elif type == self.get_constants().XRAI:
            return await self.get_probing_image_service().xrai(request)
        elif type == self.get_constants().KERNEL_SHAP:
            return await self.get_probing_image_service().kernel_shap(request)

        elif type == self.get_constants().LRP_EPSILON:
            return await self.get_lrp_service().lrp_epsilon(request)
        elif type == self.get_constants().LRP_GAMMA:
            return await self.get_lrp_service().lrp_gamma(request)
        elif type == self.get_constants().LRP_ALPHA_BETA:
            return await self.get_lrp_service().lrp_alpha_beta(request)
        elif type == self.get_constants().LRP_ZPLUS:
            return await self.get_lrp_service().lrp_zplus(request)
        elif type == self.get_constants().LRP_W2:
            return await self.get_lrp_service().lrp_w2(request)

        return HeatmapResponse(success=True)

    async def generate_heatmap_statistics(self, request: HeatmapStatisticRequest) -> HeatmapStatisticResponse:
        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=request.images,
            overwrite=request.overwrite,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        model_info: ModelInfo = await self.model_service.load_model(request.model)

        softmax_scores = []

        for image in request.images:
            list_of_scores = []
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            image_from_dataset_path = os.path.join(
                self.datasets_service.datasets_path, request.dataset, label, filename
            )
            image_from_heatmap_path = os.path.join(
                self.container.models_datasets_path,
                await self.utilities_service.models_datasets_path(request.model, request.dataset),
                request.heatmap,
                label,
                filename,
            )

            image_from_dataset = await self.datasets_service.load_image(image_from_dataset_path, model_info.input_shape)

            image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

            for threshold in self.constants.STATISTICS_PERCENTAGES_STEP:
                normalized_threshold = threshold / 100 * 255
                normalized_threshold = int(normalized_threshold)
                if request.threshold_type == "down":
                    image_from_heatmap_copy = np.where(image_from_heatmap < normalized_threshold, 0, 1)
                else:
                    image_from_heatmap_copy = np.where(image_from_heatmap > normalized_threshold, 0, 1)

                image_from_dataset_copy = np.copy(image_from_dataset)
                img_to_model = image_from_dataset_copy * image_from_heatmap_copy
                img_to_model = img_to_model.astype(np.uint8)
                img_tensor = await self.utilities_service.convert_image_to_tensor(img_to_model, grad=True)
                model_info.model.eval()

                output = model_info.model(img_tensor)
                output_softmax = torch.softmax(output, dim=1)
                list_of_scores.append(output_softmax[0, target].item())

            softmax_scores.append(ListOfScores(scores=list_of_scores))

        return HeatmapStatisticResponse(success=False, softmax_scores=softmax_scores)

    async def generate_heatmap_statistics_insertion_deletion(
        self,
        request: InsertionDeletionRequest,
    ) -> InsertionDeletionResponse:
        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=request.images,
            overwrite=False,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        model_info: ModelInfo = await self.model_service.load_model(request.model)
        model_info.model.eval()

        insertion_scores_all = []
        deletion_scores_all = []

        for image in request.images:
            insertion_scores = []
            deletion_scores = []
            label = image.label
            filename = image.filename
            target = image.target
            if target is None:
                target = await self.utilities_service.fill_target(model_info, label)

            image_from_dataset_path = os.path.join(
                self.datasets_service.datasets_path, request.dataset, label, filename
            )
            image_from_heatmap_path = os.path.join(
                self.container.models_datasets_path,
                await self.utilities_service.models_datasets_path(request.model, request.dataset),
                request.heatmap,
                label,
                filename,
            )

            image_from_dataset = await self.datasets_service.load_image(image_from_dataset_path, model_info.input_shape)
            image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

            for threshold in self.constants.STATISTICS_PERCENTAGES_STEP:
                normalized_threshold = threshold / 100 * 255
                normalized_threshold = int(normalized_threshold)

                image_from_heatmap_deletion = np.where(image_from_heatmap < normalized_threshold, 0, 1)
                image_from_heatmap_insertion = np.where(image_from_heatmap > normalized_threshold, 0, 1)

                image_from_dataset_copy_deletion = np.copy(image_from_dataset)
                img_to_model_deletion = image_from_dataset_copy_deletion * image_from_heatmap_deletion
                img_to_model_deletion = img_to_model_deletion.astype(np.uint8)
                img_tensor_deletion = await self.utilities_service.convert_image_to_tensor(
                    img_to_model_deletion, grad=False
                )

                image_from_dataset_copy_insertion = np.copy(image_from_dataset)
                img_to_model_insertion = image_from_dataset_copy_insertion * image_from_heatmap_insertion
                img_to_model_insertion = img_to_model_insertion.astype(np.uint8)
                img_tensor_insertion = await self.utilities_service.convert_image_to_tensor(
                    img_to_model_insertion, grad=False
                )

                with torch.no_grad():
                    output_deletion = model_info.model(img_tensor_deletion)
                    output_softmax_deletion = torch.softmax(output_deletion, dim=1)
                    deletion_scores.append(output_softmax_deletion[0, target].item())

                    output_insertion = model_info.model(img_tensor_insertion)
                    output_softmax_insertion = torch.softmax(output_insertion, dim=1)
                    insertion_scores.append(output_softmax_insertion[0, target].item())

            insertion_scores_all.append(ListOfScores(scores=insertion_scores))
            deletion_scores_all.append(ListOfScores(scores=deletion_scores))

        insertion_auc = self._calculate_auc(insertion_scores_all)
        deletion_auc = self._calculate_auc(deletion_scores_all)

        return InsertionDeletionResponse(
            insertion_scores=insertion_scores_all,
            deletion_scores=deletion_scores_all,
            insertion_auc=insertion_auc,
            deletion_auc=deletion_auc,
        )

    def _calculate_auc(self, scores_list: list[ListOfScores]) -> float:
        if not scores_list or not scores_list[0].scores:
            return 0.0

        num_points = len(scores_list[0].scores)
        if num_points == 0:
            return 0.0

        if num_points == 1:
            x_data = [0]
        else:
            step = 100 / (num_points - 1)
            x_data = [i * step for i in range(num_points)]

        total_auc = 0.0
        for scores_obj in scores_list:
            y_data = scores_obj.scores
            auc = 0.0
            for i in range(1, len(y_data)):
                auc += 0.5 * (y_data[i] + y_data[i - 1]) * (x_data[i] - x_data[i - 1])
            total_auc += auc / 100

        return total_auc / len(scores_list)

    async def generate_heatmap_statistics_dataset(
        self, request: HeatmapStatisticsDatasetRequest
    ) -> HeatmapStatisticsDatasetResponse:
        images = [
            LabelFilenameTarget(label=label, filename=filename, target=None)
            for label in await self.datasets_service.list_labels_in_dataset(request.dataset)
            for filename in await self.datasets_service.list_filenames_in_label(request.dataset, label)
        ]

        images = random.sample(images, int(len(images) * (request.percentage / 100)))

        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=images,
            overwrite=True,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        model_info: ModelInfo = await self.model_service.load_model(request.model)

        model_info.model.eval()

        normalized_threshold = int((request.threshold / 100) * 255)
        models_datasets_path = await self.utilities_service.models_datasets_path(request.model, request.dataset)
        heatmap_base_path = os.path.join(self.container.models_datasets_path, models_datasets_path, request.heatmap)

        original_softmax_scores = []
        modified_softmax_scores = []

        with torch.no_grad():
            for image_info in images:
                label = image_info.label
                filename = image_info.filename

                target = await self.utilities_service.fill_target(model_info, label)

                image_from_dataset_path = os.path.join(
                    self.datasets_service.datasets_path, request.dataset, label, filename
                )
                image_from_heatmap_path = os.path.join(heatmap_base_path, label, filename)

                image_from_dataset = await self.datasets_service.load_image(
                    image_from_dataset_path, model_info.input_shape
                )

                image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

                mask = np.where(image_from_heatmap < normalized_threshold, 0, 1)

                img_to_model = (image_from_dataset * mask).astype(np.uint8)

                img_tensor_modified = await self.utilities_service.convert_image_to_tensor(img_to_model, grad=False)
                img_tensor_original = await self.utilities_service.convert_image_to_tensor(
                    image_from_dataset, grad=False
                )

                output_modified = model_info.model(img_tensor_modified)
                output_softmax_modified = torch.softmax(output_modified, dim=1)
                score_modified = output_softmax_modified[0, target].item()
                modified_softmax_scores.append(score_modified)

                output_original = model_info.model(img_tensor_original)
                output_softmax_original = torch.softmax(output_original, dim=1)
                score_original = output_softmax_original[0, target].item()
                original_softmax_scores.append(score_original)

        rate_of_increase_in_scores = sum(
            1
            for original, modified in zip(original_softmax_scores, modified_softmax_scores, strict=False)
            if modified > original
        ) / len(original_softmax_scores)

        average_drop = sum(
            max(0, original - modified)
            for original, modified in zip(original_softmax_scores, modified_softmax_scores, strict=False)
        ) / len(original_softmax_scores)

        return HeatmapStatisticsDatasetResponse(
            success=True, rate_of_increase_in_scores=rate_of_increase_in_scores, average_drop=average_drop
        )

    async def generate_heatmap_statistics_dataset_many(
        self, request: HeatmapStatisticsDatasetManyRequest
    ) -> HeatmapStatisticsDatasetManyResponse:
        list_of_rate_of_increase_in_scores = []
        list_of_average_drop = []

        for percentage in self.constants.STATISTICS_PERCENTAGES_STEP:
            modified_request = HeatmapStatisticsDatasetRequest(
                model=request.model,
                dataset=request.dataset,
                heatmap=request.heatmap,
                threshold=percentage,
                percentage=request.percentage,
            )
            response = await self.generate_heatmap_statistics_dataset(request=modified_request)
            list_of_rate_of_increase_in_scores.append(response.rate_of_increase_in_scores)
            list_of_average_drop.append(response.average_drop)

        return HeatmapStatisticsDatasetManyResponse(
            success=True,
            list_of_rate_of_increase_in_scores=list_of_rate_of_increase_in_scores,
            list_of_average_drop=list_of_average_drop,
        )

    async def generate_heatmap_statistics_label(
        self, request: HeatmapStatisticsLabelRequest
    ) -> HeatmapStatisticsLabelResponse:
        images = [
            LabelFilenameTarget(label=request.label, filename=filename, target=None)
            for filename in await self.datasets_service.list_filenames_in_label(request.dataset, request.label)
        ]

        images = random.sample(images, int(len(images) * (request.percentage / 100)))

        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=images,
            overwrite=True,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        model_info: ModelInfo = await self.model_service.load_model(request.model)

        model_info.model.eval()

        normalized_threshold = int((request.threshold / 100) * 255)
        models_datasets_path = await self.utilities_service.models_datasets_path(request.model, request.dataset)
        heatmap_base_path = os.path.join(self.container.models_datasets_path, models_datasets_path, request.heatmap)

        original_softmax_scores = []
        modified_softmax_scores = []

        with torch.no_grad():
            for image_info in images:
                label = image_info.label
                filename = image_info.filename

                target = await self.utilities_service.fill_target(model_info, label)

                image_from_dataset_path = os.path.join(
                    self.datasets_service.datasets_path, request.dataset, label, filename
                )
                image_from_heatmap_path = os.path.join(heatmap_base_path, label, filename)

                image_from_dataset = await self.datasets_service.load_image(
                    image_from_dataset_path, model_info.input_shape
                )

                image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

                mask = np.where(image_from_heatmap < normalized_threshold, 0, 1)

                img_to_model = (image_from_dataset * mask).astype(np.uint8)

                img_tensor_modified = await self.utilities_service.convert_image_to_tensor(img_to_model, grad=False)
                img_tensor_original = await self.utilities_service.convert_image_to_tensor(
                    image_from_dataset, grad=False
                )

                output_modified = model_info.model(img_tensor_modified)
                output_softmax_modified = torch.softmax(output_modified, dim=1)
                score_modified = output_softmax_modified[0, target].item()
                modified_softmax_scores.append(score_modified)

                output_original = model_info.model(img_tensor_original)
                output_softmax_original = torch.softmax(output_original, dim=1)
                score_original = output_softmax_original[0, target].item()
                original_softmax_scores.append(score_original)

        rate_of_increase_in_scores = sum(
            1
            for original, modified in zip(original_softmax_scores, modified_softmax_scores, strict=False)
            if modified > original
        ) / len(original_softmax_scores)

        average_drop = sum(
            max(0, original - modified)
            for original, modified in zip(original_softmax_scores, modified_softmax_scores, strict=False)
        ) / len(original_softmax_scores)

        return HeatmapStatisticsLabelResponse(
            success=True, rate_of_increase_in_scores=rate_of_increase_in_scores, average_drop=average_drop
        )

    async def generate_heatmap_statistics_label_many(
        self, request: HeatmapStatisticsLabelManyRequest
    ) -> HeatmapStatisticsLabelManyResponse:
        list_of_rate_of_increase_in_scores = []
        list_of_average_drop = []

        for percentage in self.constants.STATISTICS_PERCENTAGES_STEP:
            modified_request = HeatmapStatisticsLabelRequest(
                model=request.model,
                dataset=request.dataset,
                heatmap=request.heatmap,
                threshold=percentage,
                percentage=10,
                label=request.label,
            )
            response = await self.generate_heatmap_statistics_label(request=modified_request)
            list_of_rate_of_increase_in_scores.append(response.rate_of_increase_in_scores)
            list_of_average_drop.append(response.average_drop)

        return HeatmapStatisticsLabelManyResponse(
            success=True,
            list_of_rate_of_increase_in_scores=list_of_rate_of_increase_in_scores,
            list_of_average_drop=list_of_average_drop,
        )

    async def generate_heatmap_statistics_dataset_shapes(
        self, request: ShapeMetricsDatasetRequest
    ) -> ShapeMetricsDatasetResponse:
        areas = []
        perimeters = []
        circularities = []
        rectangularities = []
        images = [
            LabelFilenameTarget(label=label, filename=filename, target=None)
            for label in await self.datasets_service.list_labels_in_dataset(request.dataset)
            for filename in await self.datasets_service.list_filenames_in_label(request.dataset, label)
        ]

        images = random.sample(images, int(len(images) * (request.percentage / 100)))

        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=images,
            overwrite=True,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        for image in images:
            label = image.label
            filename = image.filename

            image_from_heatmap_path = os.path.join(
                self.container.models_datasets_path,
                await self.utilities_service.models_datasets_path(request.model, request.dataset),
                request.heatmap,
                label,
                filename,
            )

            image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

            image_from_heatmap_normalized = image_from_heatmap / 255
            image_binary = (image_from_heatmap_normalized > self.constants.THRESHOLD_ACTIVATIONS).astype(np.uint8)

            area = np.sum(image_binary)
            areas.append(area)

            contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)

            total_perimeter = max(0, total_perimeter)
            perimeters.append(total_perimeter)

            circularity = total_perimeter**2 / (4 * math.pi * area) if area > 0 else 0
            circularities.append(circularity)

            if len(contours) > 0:
                all_points = np.vstack(list(contours))
                mbr = cv2.minAreaRect(all_points)
                mbr_area = mbr[1][0] * mbr[1][1]

                rectangularity = area / mbr_area if mbr_area > 0 else 0
            else:
                rectangularity = 0
            rectangularities.append(rectangularity)

        return ShapeMetricsDatasetResponse(
            success=True,
            areas=areas,
            perimeters=perimeters,
            circularities=circularities,
            rectangularities=rectangularities,
        )

    async def generate_heatmap_statistics_label_shapes(
        self, request: ShapeMetricsLabelRequest
    ) -> ShapeMetricsLabelResponse:
        areas = []
        perimeters = []
        circularities = []
        rectangularities = []

        images = [
            LabelFilenameTarget(label=request.label, filename=filename, target=None)
            for filename in await self.datasets_service.list_filenames_in_label(request.dataset, request.label)
        ]

        images = random.sample(images, int(len(images) * (request.percentage / 100)))

        base_request: HeatmapRequest = HeatmapRequest(
            model=request.model,
            dataset=request.dataset,
            images=images,
            overwrite=True,
        )

        await self.generate_heatmap(request=base_request, type=request.heatmap)

        for image in images:
            label = image.label
            filename = image.filename

            image_from_heatmap_path = os.path.join(
                self.container.models_datasets_path,
                await self.utilities_service.models_datasets_path(request.model, request.dataset),
                request.heatmap,
                label,
                filename,
            )

            image_from_heatmap = np.array(Image.open(image_from_heatmap_path))

            image_from_heatmap = image_from_heatmap / 255

            image_binary = image_from_heatmap > self.constants.THRESHOLD_ACTIVATIONS

            area = np.sum(image_binary)

            areas.append(area)

            image_binary_uint8 = image_binary.astype(np.uint8)
            contours, _ = cv2.findContours(image_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)

            total_perimeter = max(0, total_perimeter)
            perimeters.append(total_perimeter)

            circularity = total_perimeter**2 / (4 * math.pi * area) if area > 0 else 0
            circularities.append(circularity)

            if len(contours) > 0:
                all_points = np.vstack(list(contours))
                mbr = cv2.minAreaRect(all_points)
                mbr_area = mbr[1][0] * mbr[1][1]

                rectangularity = area / mbr_area if mbr_area > 0 else 0
            else:
                rectangularity = 0
            rectangularities.append(rectangularity)

        return ShapeMetricsLabelResponse(
            success=True,
            areas=areas,
            perimeters=perimeters,
            circularities=circularities,
            rectangularities=rectangularities,
        )
