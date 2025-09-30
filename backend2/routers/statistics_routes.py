from fastapi import APIRouter, Depends

from services.container import StatisticsService, get_statistics_service
from interfaces.all_quantitative_statistics import AllQuantitativeStatisticsRequest, AllQuantitativeStatisticsResponse

router = APIRouter()


@router.post('/all-quantitative-statistics')
async def all_quantitative_statistics(request: AllQuantitativeStatisticsRequest,
                                      statistics_service: StatisticsService = Depends(
                                          get_statistics_service)
                                      ):
    response = await statistics_service.all_quantitative_statistics(request)
    return response
