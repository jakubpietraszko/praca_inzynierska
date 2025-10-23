from fastapi import APIRouter, Depends

from interfaces.unsupervised_activations import (
    UnsupervisedActivationsRequest,
    UnsupervisedActivationsResponse,
)
from interfaces.unsupervised_heatmaps import UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
from services.container import (
    UnsupervisedService,
    get_unsupervised_service,
)

router = APIRouter()


@router.post("/activations", response_model=UnsupervisedActivationsResponse)
async def activations(
    request: UnsupervisedActivationsRequest,
    unsupervised_service: UnsupervisedService = Depends(get_unsupervised_service),  # noqa: B008
):
    return await unsupervised_service.activations(request)


@router.post("/heatmaps", response_model=UnsupervisedHeatmapsResponse)
async def heatmaps(
    request: UnsupervisedHeatmapsRequest,
    unsupervised_service: UnsupervisedService = Depends(get_unsupervised_service),  # noqa: B008
):
    return await unsupervised_service.heatmaps(request)
