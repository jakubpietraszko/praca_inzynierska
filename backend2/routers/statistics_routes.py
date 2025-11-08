import re

from fastapi import APIRouter, Depends

from interfaces.all_quantitative_statistics import AllQuantitativeStatisticsRequest, AllQuantitativeStatisticsResponse
from interfaces.heatmaps_statistics_dataset import HeatmapsStatisticsDatasetRequest, HeatmapsStatisticsDatasetResponse
from interfaces.heatmaps_statistics_image import HeatmapsStatisticsImageRequest, HeatmapsStatisticsImageResponse
from interfaces.model_dataset_statistics import ModelDatasetStatisticsRequest, ModelDatasetStatisticsResponse
from services.container import StatisticsService, get_statistics_service

router = APIRouter()


@router.post("/all-quantitative-statistics", response_model=AllQuantitativeStatisticsResponse)
async def all_quantitative_statistics(
    request: AllQuantitativeStatisticsRequest,
    statistics_service: StatisticsService = Depends(get_statistics_service),  # noqa: B008
):
    response = await statistics_service.all_quantitative_statistics(request)
    return response


@router.post("/heatmaps-statistics-image", response_model=HeatmapsStatisticsImageResponse)
async def heatmaps_statistics_image(
    request: HeatmapsStatisticsImageRequest,
    statistics_service: StatisticsService = Depends(get_statistics_service),  # noqa: B008
):
    response = await statistics_service.heatmaps_statistics_image(request)
    return response


@router.post("/heatmaps-statistics-dataset", response_model=HeatmapsStatisticsDatasetResponse)
async def heatmaps_statistics_dataset(
    request: HeatmapsStatisticsDatasetRequest,
    statistics_service: StatisticsService = Depends(get_statistics_service),  # noqa: B008
):
    response = await statistics_service.heatmaps_statistics_dataset(request)
    return response

@router.post('/model-dataset-statistics', response_model=ModelDatasetStatisticsResponse)
async def model_dataset_statistics(
    request: ModelDatasetStatisticsRequest,
    statistics_service: StatisticsService = Depends(get_statistics_service),  # noqa: B008
) -> ModelDatasetStatisticsResponse:
    response: ModelDatasetStatisticsResponse = await statistics_service.generate_model_dataset_statistics_with_cache(request)
    return response
