from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import HeatmapsFacedeService, get_constants, get_heatmaps_facede_service

router = APIRouter()


@router.post("/grad-cam", response_model=GradientResponse)
async def grad_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRAD_CAM,
    )


@router.post("/grad-cam-plus-plus", response_model=GradientResponse)
async def grad_cam_plus(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRAD_CAM_PLUS_PLUS,
    )


@router.post("/group-cam", response_model=GradientResponse)
async def group_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GROUP_CAM,
    )


@router.post("/hires-cam", response_model=GradientResponse)
async def hires_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.HIRES_CAM,
    )


@router.post("/wgrad-cam", response_model=GradientResponse)
async def wgrad_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.WGRAD_CAM,
    )


@router.post("/ablation-cam", response_model=GradientResponse)
async def ablation_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.ABLATION_CAM,
    )


@router.post("/xgrad-cam", response_model=GradientResponse)
async def xgrad_cam(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.XGRAD_CAM,
    )
