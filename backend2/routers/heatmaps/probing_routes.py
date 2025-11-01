from fastapi import APIRouter, Depends

from interfaces.heatmap import HeatmapRequest, HeatmapResponse
from services.container import Constants, HeatmapsFacedeService, get_constants, get_heatmaps_facede_service

router = APIRouter()


@router.post("/lime", response_model=HeatmapResponse)
async def lime(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.LIME,
    )


@router.post("/xrai", response_model=HeatmapResponse)
async def xrai(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.XRAI,
    )


@router.post("/kernel-shap", response_model=HeatmapResponse)
async def kernel_shap(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.KERNEL_SHAP,
    )
