from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import get_constants, get_heatmaps_facede_service
from services.heatmaps_facede_service import HeatmapsFacedeService

router = APIRouter()


@router.post("/lrp-epsilon", response_model=GradientResponse)
async def gradient(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_EPSILON,
    )


@router.post("/lrp-gamma", response_model=GradientResponse)
async def lrp_gamma(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_GAMMA,
    )


@router.post("/lrp-alpha-beta", response_model=GradientResponse)
async def lrp_alpha_beta(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_ALPHA_BETA,
    )


@router.post("/lrp-zplus", response_model=GradientResponse)
async def lrp_zplus(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_ZPLUS,
    )


@router.post("/lrp-w2", response_model=GradientResponse)
async def lrp_w2(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LRP_W2,
    )
