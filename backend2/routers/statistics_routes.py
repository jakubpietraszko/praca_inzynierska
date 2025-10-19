from fastapi import APIRouter, Depends

from interfaces.all_quantitative_statistics import AllQuantitativeStatisticsRequest, AllQuantitativeStatisticsResponse
from interfaces.heatmaps_statistics_dataset import HeatmapsStatisticsDatasetRequest, HeatmapsStatisticsDatasetResponse
from interfaces.heatmaps_statistics_image import HeatmapsStatisticsImageRequest, HeatmapsStatisticsImageResponse
from services.container import StatisticsService, get_statistics_service

router = APIRouter()


@router.post('/all-quantitative-statistics', response_model=AllQuantitativeStatisticsResponse)
async def all_quantitative_statistics(request: AllQuantitativeStatisticsRequest):
    statistics_service: StatisticsService = Depends(get_statistics_service)
    response = await statistics_service.all_quantitative_statistics(request)
    return response


@router.post('/heatmaps-statistics-image', response_model=HeatmapsStatisticsImageResponse)
async def heatmaps_statistics_image(request: HeatmapsStatisticsImageRequest):
    statistics_service: StatisticsService = Depends(get_statistics_service)
    response = await statistics_service.heatmaps_statistics_image(request)
    return response


@router.post('/heatmaps-statistics-dataset', response_model=HeatmapsStatisticsDatasetResponse)
async def heatmaps_statistics_dataset(request: HeatmapsStatisticsDatasetRequest):
    statistics_service: StatisticsService = Depends(get_statistics_service)
    response = await statistics_service.heatmaps_statistics_dataset(request)
    return response
