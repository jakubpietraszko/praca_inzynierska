import torch
from torch.utils.data import DataLoader

from interfaces.all_quantitative_statistics import AllQuantitativeStatisticsRequest, AllQuantitativeStatisticsResponse
from interfaces.gradient import GradientRequest, LabelFilenameTarget
from interfaces.heatmaps_statistics_dataset import HeatmapsStatisticsDatasetRequest, HeatmapsStatisticsDatasetResponse
from interfaces.heatmaps_statistics_image import HeatmapsStatisticsImageRequest, HeatmapsStatisticsImageResponse


class StatisticsService:
    def __init__(self):
        from services.container import (
            Constants,
            DatasetsService,
            HeatmapsFacedeService,
            ModelsDatasetsService,
            ModelsService,
            get_constants,
            get_datasets_service,
            get_heatmaps_facede_service,
            get_models_datasets_service,
            get_models_service,
        )

        self.models_service: ModelsService = get_models_service()
        self.datasets_service: DatasetsService = get_datasets_service()
        self.models_datasets_service: ModelsDatasetsService = get_models_datasets_service()
        self.heatmaps_facede_service: HeatmapsFacedeService = get_heatmaps_facede_service()
        self.constants: Constants = get_constants()

    async def generate_all_quantitative_statistics(
        self, request: AllQuantitativeStatisticsRequest
    ) -> AllQuantitativeStatisticsResponse:
        model_info = await self.models_service.load_model(request.model)
        labels = await self.datasets_service.list_labels_in_dataset(request.dataset)
        label_to_idx = model_info.label_to_idx

        dataset_info = await self.datasets_service.load_dataset(request.dataset, model_info.transform)
        dataset = dataset_info.dataset
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        label_to_idx = model_info.label_to_idx
        idx_to_label = {v: k for k, v in label_to_idx.items()}

        confusion_dict = {label: dict.fromkeys(labels, 0) for label in labels}

        model = model_info.model

        model.eval()
        dataloader = DataLoader(dataset, batch_size=self.constants.BATCH_SIZE, shuffle=False)

        device = next(model.parameters()).device if next(model.parameters(), None) is not None else "cpu"
        model.to(device)

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, 1)
                for true_idx, pred_idx in zip(targets, preds, strict=False):
                    true_idx = true_idx.item()
                    pred_idx = pred_idx.item()

                    true_label = idx_to_class[true_idx]
                    pred_label = idx_to_label[pred_idx]
                    true_label = idx_to_class[true_idx]
                    pred_label = idx_to_label[pred_idx]
                    confusion_dict[true_label][pred_label] += 1

        return AllQuantitativeStatisticsResponse(success=True, confusion_dict=confusion_dict)

    async def all_quantitative_statistics(
        self, request: AllQuantitativeStatisticsRequest
    ) -> AllQuantitativeStatisticsResponse:
        response = await self.generate_all_quantitative_statistics(request)
        return response

    async def deletion_score(self):
        pass

    async def insertion_score(self):
        pass

    async def overall_score(self):
        pass

    async def generate_deletion_plot(self):
        pass

    async def generate_insertion_plot(self):
        pass

    async def heatmaps_statistics_image(
        self, request: HeatmapsStatisticsImageRequest
    ) -> HeatmapsStatisticsImageResponse:
        await self.heatmaps_facede_service.generate_heatmap(
            GradientRequest(
                model=request.model,
                dataset=request.dataset,
                images=[LabelFilenameTarget(label=request.label, filename=request.image, target=None)],
                overwrite=False,
            ),
            type=request.heatmap,
        )

        image = await self.models_datasets_service.load_image(
            model=request.model,
            dataset=request.dataset,
            label=request.label,
            heatmap=request.heatmap,
            filename=request.image,
        )

        print(image.shape)

        return HeatmapsStatisticsImageResponse(
            deletion=0.0,
            insertion=0.0,
            overall=0.0,
            success=True,
        )

    async def heatmaps_statistics_dataset(
        self, request: HeatmapsStatisticsDatasetRequest
    ) -> HeatmapsStatisticsDatasetResponse:
        return HeatmapsStatisticsDatasetResponse(
            average_drop=[0.0, 0.33, 0.66, 1.0], rate_of_increase=[0.0, 0.33, 0.66, 1.0], success=True
        )
