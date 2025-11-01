from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.container import get_constants, get_heatmaps_facede_service
from services.heatmaps_facede_service import HeatmapsFacedeService

router = APIRouter()


@router.post("/gradient", response_model=HeatmapResponse)
async def gradient(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT,
    )


@router.post("/gradient-squared", response_model=HeatmapResponse)
async def gradient_squared(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT_SQUARED,
    )


@router.post("/deconvolution", response_model=HeatmapResponse)
async def deconvolution(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.DECONVOLUTION,
    )


@router.post("/guided-backpropagation", response_model=HeatmapResponse)
async def guided_backpropagation(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GUIDED_BACKPROPAGATION,
    )


@router.post("/input-gradient", response_model=HeatmapResponse)
async def input_gradient(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.INPUT_GRADIENT,
    )
