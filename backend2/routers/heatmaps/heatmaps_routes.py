import os

from fastapi import APIRouter, Depends
from interfaces.gradient import GradientRequest, GradientResponse
from interfaces.all_heatmaps import AllHeatmapsResponse
from constans import Constants
from services.container import get_constants

from services.container import HeatmapsService, get_heatmaps_service
from services.container import HeatmapsFacedeService, get_heatmaps_facede_service

router = APIRouter()


@router.get('/all-heatmaps', response_model=AllHeatmapsResponse)
async def all_available_heatmaps(
    constants: Constants = Depends(
        get_constants
    ),
    heatmaps_service: HeatmapsService = Depends(
        get_heatmaps_service
    )
):
    response = heatmaps_service.all_available_heatmaps(
        constants=constants
    )

    return response


@router.post('/all-heatmaps', response_model=GradientResponse)
async def get_all_heatmaps(request: GradientRequest,
                           constants: Constants = Depends(get_constants),
                           heatmaps_facede_service: HeatmapsFacedeService = Depends(
                               get_heatmaps_facede_service
                           )
                           ):
    responses = list()

    for type in constants.AVAILABLE_HEATMAPS:
        responses.append(
            heatmaps_facede_service.generate_heatmap(
                request=request,
                type=type,
            )
        )

    return GradientResponse(
        success=True,
    )
