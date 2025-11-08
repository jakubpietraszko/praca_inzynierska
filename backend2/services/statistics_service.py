import os

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from interfaces.all_quantitative_statistics import AllQuantitativeStatisticsRequest, AllQuantitativeStatisticsResponse
from interfaces.heatmaps_statistics_dataset import HeatmapsStatisticsDatasetRequest, HeatmapsStatisticsDatasetResponse
from interfaces.model_dataset_statistics import (
    ModelDatasetPredictionsRequest,
    ModelDatasetPredictionsResponse,
    ModelDatasetStatisticsRequest,
    ModelDatasetStatisticsResponse,
)
from interfaces.proba_for_image import ProbaForImageRequest, ProbaForImageResponse


class StatisticsService:
    def __init__(self) -> None:
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

    async def heatmaps_statistics_dataset(
        self, request: HeatmapsStatisticsDatasetRequest
    ) -> HeatmapsStatisticsDatasetResponse:
        return HeatmapsStatisticsDatasetResponse(
            average_drop=[0.0, 0.33, 0.66, 1.0], rate_of_increase=[0.0, 0.33, 0.66, 1.0], success=True
        )

    async def generate_model_dataset_predictions(
        self, request: ModelDatasetPredictionsRequest
    ) -> ModelDatasetPredictionsResponse:
        path = await self.models_datasets_service.get_model_dataset_predictions_path(request.model, request.dataset)

        if os.path.exists(path):
            import json

            with open(path) as f:
                data = json.load(f)
            return ModelDatasetPredictionsResponse(
                predictions=data["predictions"],
                true_labels=data["true_labels"],
                labels=data["labels"],
                predictions_proba=data["predictions_proba"],
                confusion_matrix=data["confusion_matrix"],
                filenames=data.get("filenames", []),
            )

        model_info = await self.models_service.load_model(request.model)
        labels = await self.datasets_service.list_labels_in_dataset(request.dataset)
        label_to_idx = model_info.label_to_idx

        dataset_info = await self.datasets_service.load_dataset(request.dataset, model_info.transform)
        dataset = dataset_info.dataset
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        label_to_idx = model_info.label_to_idx
        idx_to_label = {v: k for k, v in label_to_idx.items()}

        model = model_info.model

        model.eval()
        dataloader = DataLoader(dataset, batch_size=self.constants.BATCH_SIZE, shuffle=False)

        device = next(model.parameters()).device if next(model.parameters(), None) is not None else "cpu"
        model.to(device)

        res_confusion_matrix = {label: dict.fromkeys(labels, 0) for label in labels}
        res_predictions_proba = []
        res_predictions = []
        res_true_labels = []
        res_labels = []
        res_filenames = []
        for label in labels:
            res_labels.append(label)

        label_to_common_idx = {label: i for i, label in enumerate(res_labels)}

        sample_idx = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, 1)
                for i in range(len(targets)):
                    dataset_idx = targets[i].item()
                    model_pred_idx = preds[i].item()

                    true_label = idx_to_class[dataset_idx]
                    pred_label = idx_to_label[model_pred_idx]

                    true_common_idx = label_to_common_idx[true_label]
                    pred_common_idx = label_to_common_idx[pred_label]

                    image_path = dataset.samples[sample_idx][0]
                    filename = os.path.basename(image_path)

                    probs_list = probs[i].cpu().tolist()
                    reordered_probs = [probs_list[label_to_idx[label]] for label in res_labels]

                    res_true_labels.append(true_common_idx)
                    res_predictions.append(pred_common_idx)
                    res_predictions_proba.append(reordered_probs)
                    res_filenames.append(filename)
                    res_confusion_matrix[true_label][pred_label] += 1

                    sample_idx += 1

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            import json

            json.dump(
                {
                    "predictions": res_predictions,
                    "true_labels": res_true_labels,
                    "labels": res_labels,
                    "predictions_proba": res_predictions_proba,
                    "confusion_matrix": res_confusion_matrix,
                    "filenames": res_filenames,
                },
                f,
            )

        return ModelDatasetPredictionsResponse(
            predictions_proba=res_predictions_proba,
            predictions=res_predictions,
            true_labels=res_true_labels,
            labels=res_labels,
            confusion_matrix=res_confusion_matrix,
            filenames=res_filenames,
        )

    async def generate_model_dataset_statistics_with_cache(
        self, request: ModelDatasetStatisticsRequest
    ) -> ModelDatasetStatisticsResponse:
        req = ModelDatasetPredictionsRequest(model=request.model, dataset=request.dataset)
        predictions_response: ModelDatasetPredictionsResponse = await self.generate_model_dataset_predictions(req)

        import numpy as np
        from sklearn.preprocessing import label_binarize

        labels = predictions_response.labels
        n_classes = len(labels)

        precision_per_class_arr = metrics.precision_score(
            predictions_response.true_labels, predictions_response.predictions, average=None, zero_division=0
        )

        recall_per_class_arr = metrics.recall_score(
            predictions_response.true_labels, predictions_response.predictions, average=None, zero_division=0
        )

        f1_score_per_class_arr = metrics.f1_score(
            predictions_response.true_labels, predictions_response.predictions, average=None, zero_division=0
        )

        jaccard_per_class_arr = metrics.jaccard_score(
            predictions_response.true_labels, predictions_response.predictions, average=None, zero_division=0
        )

        y_true_binarized = label_binarize(predictions_response.true_labels, classes=list(range(n_classes)))
        if n_classes == 2:
            y_true_binarized_array = np.asarray(y_true_binarized)
            y_true_binarized = np.column_stack([1 - y_true_binarized_array, y_true_binarized_array])
        else:
            y_true_binarized = np.asarray(y_true_binarized)

        y_pred_proba = np.array(predictions_response.predictions_proba)

        roc_auc_per_class_dict = {}
        for i in range(n_classes):
            try:
                y_true_class = y_true_binarized[:, i] if y_true_binarized.ndim > 1 else y_true_binarized
                roc_auc_per_class_dict[labels[i]] = float(metrics.roc_auc_score(y_true_class, y_pred_proba[:, i]))
            except (ValueError, IndexError):
                roc_auc_per_class_dict[labels[i]] = 0.0

        unique, counts = np.unique(predictions_response.true_labels, return_counts=True)
        support_dict = {labels[idx]: int(count) for idx, count in zip(unique, counts, strict=False)}
        for label in labels:
            if label not in support_dict:
                support_dict[label] = 0

        precision_per_class_dict = {labels[i]: float(np.asarray(precision_per_class_arr)[i]) for i in range(n_classes)}
        recall_per_class_dict = {labels[i]: float(np.asarray(recall_per_class_arr)[i]) for i in range(n_classes)}
        f1_score_per_class_dict = {labels[i]: float(np.asarray(f1_score_per_class_arr)[i]) for i in range(n_classes)}
        jaccard_per_class_dict = {labels[i]: float(np.asarray(jaccard_per_class_arr)[i]) for i in range(n_classes)}

        if n_classes == 2:
            roc_auc_macro = float(metrics.roc_auc_score(predictions_response.true_labels, y_pred_proba[:, 1]))
            roc_auc_micro = float(metrics.roc_auc_score(predictions_response.true_labels, y_pred_proba[:, 1]))
            roc_auc_weighted = float(metrics.roc_auc_score(predictions_response.true_labels, y_pred_proba[:, 1]))
        else:
            roc_auc_macro = float(
                metrics.roc_auc_score(
                    predictions_response.true_labels,
                    predictions_response.predictions_proba,
                    average="macro",
                    multi_class="ovr",
                )
            )
            roc_auc_micro = float(
                metrics.roc_auc_score(
                    predictions_response.true_labels,
                    predictions_response.predictions_proba,
                    average="micro",
                    multi_class="ovr",
                )
            )
            roc_auc_weighted = float(
                metrics.roc_auc_score(
                    predictions_response.true_labels,
                    predictions_response.predictions_proba,
                    average="weighted",
                    multi_class="ovr",
                )
            )

        return ModelDatasetStatisticsResponse(
            accuracy=float(metrics.accuracy_score(predictions_response.true_labels, predictions_response.predictions)),
            balanced_accuracy=float(
                metrics.balanced_accuracy_score(predictions_response.true_labels, predictions_response.predictions)
            ),
            confusion_matrix=predictions_response.confusion_matrix,
            roc_auc_per_class=roc_auc_per_class_dict,
            roc_auc_macro=roc_auc_macro,
            roc_auc_micro=roc_auc_micro,
            roc_auc_weighted=roc_auc_weighted,
            support_per_class=support_dict,
            precision_per_class=precision_per_class_dict,
            precision_macro=float(
                metrics.precision_score(
                    predictions_response.true_labels, predictions_response.predictions, average="macro", zero_division=0
                )
            ),
            precision_micro=float(
                metrics.precision_score(
                    predictions_response.true_labels, predictions_response.predictions, average="micro", zero_division=0
                )
            ),
            precision_weighted=float(
                metrics.precision_score(
                    predictions_response.true_labels,
                    predictions_response.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            recall_per_class=recall_per_class_dict,
            recall_macro=float(
                metrics.recall_score(
                    predictions_response.true_labels, predictions_response.predictions, average="macro", zero_division=0
                )
            ),
            recall_micro=float(
                metrics.recall_score(
                    predictions_response.true_labels, predictions_response.predictions, average="micro", zero_division=0
                )
            ),
            recall_weighted=float(
                metrics.recall_score(
                    predictions_response.true_labels,
                    predictions_response.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            f1_score_per_class=f1_score_per_class_dict,
            f1_score_macro=float(
                metrics.f1_score(
                    predictions_response.true_labels, predictions_response.predictions, average="macro", zero_division=0
                )
            ),
            f1_score_micro=float(
                metrics.f1_score(
                    predictions_response.true_labels, predictions_response.predictions, average="micro", zero_division=0
                )
            ),
            f1_score_weighted=float(
                metrics.f1_score(
                    predictions_response.true_labels,
                    predictions_response.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            jaccard_per_class=jaccard_per_class_dict,
            jaccard_macro=float(
                metrics.jaccard_score(
                    predictions_response.true_labels, predictions_response.predictions, average="macro", zero_division=0
                )
            ),
            jaccard_micro=float(
                metrics.jaccard_score(
                    predictions_response.true_labels, predictions_response.predictions, average="micro", zero_division=0
                )
            ),
            jaccard_weighted=float(
                metrics.jaccard_score(
                    predictions_response.true_labels,
                    predictions_response.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        )

    async def proba_for_image(self, request: ProbaForImageRequest) -> ProbaForImageResponse:
        req: ModelDatasetPredictionsRequest = ModelDatasetPredictionsRequest(
            model=request.model,
            dataset=request.dataset,
        )
        res: ModelDatasetPredictionsResponse = await self.generate_model_dataset_predictions(req)

        try:
            image_idx = res.filenames.index(request.filename)
        except ValueError as exc:
            raise ValueError(f"Image with filename '{request.filename}' not found in dataset predictions") from exc

        probabilities = res.predictions_proba[image_idx]
        classes = res.labels

        return ProbaForImageResponse(probabilities=probabilities, classes=classes)
