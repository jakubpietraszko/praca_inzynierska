from fastapi import APIRouter, Depends

from interfaces.gradient import GradientRequest, GradientResponse
from services.container import get_constants
from services.heatmaps_facede_service import HeatmapsFacedeService
from services.container import get_heatmaps_facede_service
from constans import Constants


router = APIRouter()


@router.post('/gradient', response_model=GradientResponse)
async def gradient(request: GradientRequest,
                   constants: Constants = Depends(get_constants),
                   heatmaps_facede_service: HeatmapsFacedeService = Depends(
                       get_heatmaps_facede_service
                   )
                   ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT,
    )


@router.post('/gradient-squared', response_model=GradientResponse)
async def gradient_squared(request: GradientRequest,
                           constants: Constants = Depends(get_constants),
                           heatmaps_facede_service: HeatmapsFacedeService = Depends(
                               get_heatmaps_facede_service
                           )

                           ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GRADIENT_SQUARED,
    )


@router.post('/deconvolution', response_model=GradientResponse)
async def deconvolution(request: GradientRequest,
                        constants: Constants = Depends(get_constants),
                        heatmaps_facede_service: HeatmapsFacedeService = Depends(
                            get_heatmaps_facede_service
                        )
                        ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.DECONVOLUTION,
    )


@router.post('/guided-backpropagation', response_model=GradientResponse)
async def guided_backpropagation(request: GradientRequest,
                                 constants: Constants = Depends(get_constants),
                                 heatmaps_facede_service: HeatmapsFacedeService = Depends(
                                     get_heatmaps_facede_service
                                 )
                                 ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.GUIDED_BACKPROPAGATION,
    )


@router.post('/input-gradient', response_model=GradientResponse)
async def input_gradient(request: GradientRequest,
                         constants: Constants = Depends(get_constants),
                         heatmaps_facede_service: HeatmapsFacedeService = Depends(
                             get_heatmaps_facede_service
                         )
                         ):
    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=constants.INPUT_GRADIENT,
    )
