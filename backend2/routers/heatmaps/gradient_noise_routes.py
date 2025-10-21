from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import HeatmapsFacedeService, get_constants, get_heatmaps_facede_service

router = APIRouter()


@router.post("/smooth-gradient", response_model=GradientResponse)
async def smooth_grad(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.SMOOTH_GRADIENT,
    )


@router.post("/smooth-gradient-square", response_model=GradientResponse)
async def smooth_grad_square(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.SMOOTH_GRADIENT_SQUARE,
    )


@router.post("/var-gradient", response_model=GradientResponse)
async def var_grad(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.VAR_GRADIENT,
    )
