from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import HeatmapsFacedeService, get_constants, get_heatmaps_facede_service

router = APIRouter()


@router.post('/integrated-gradient', response_model=GradientResponse)
async def integrated_gradient(request: GradientRequest,
                              constants: Constants = Depends(get_constants),
                              heatmaps_facede_service: HeatmapsFacedeService = Depends(
                                  get_heatmaps_facede_service
                              ),
                              ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.INTEGRATED_GRADIENT,
    )
