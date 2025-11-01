import os

from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.all_heatmaps import AllHeatmapsResponse
from interfaces.heatmap import (
    HeatmapRequest,
    HeatmapResponse,
    HeatmapStatisticRequest,
    HeatmapStatisticResponse,
    HeatmapStatisticsDatasetManyRequest,
    HeatmapStatisticsDatasetManyResponse,
    HeatmapStatisticsDatasetRequest,
    HeatmapStatisticsDatasetResponse,
    HeatmapStatisticsLabelManyRequest,
    HeatmapStatisticsLabelManyResponse,
    HeatmapStatisticsLabelRequest,
    HeatmapStatisticsLabelResponse,
    LabelFilenameTarget,
)
from interfaces.shapes import (
    ShapeMetricsDatasetRequest,
    ShapeMetricsDatasetResponse,
    ShapeMetricsLabelRequest,
    ShapeMetricsLabelResponse,
)
from interfaces.unsupervised_heatmaps import UnsupervisedHeatmapsRequest, UnsupervisedHeatmapsResponse
from services.container import (
    HeatmapsFacedeService,
    HeatmapsService,
    get_constants,
    get_container,
    get_heatmaps_facede_service,
    get_heatmaps_service,
)

router = APIRouter()


@router.get("/all-heatmaps", response_model=AllHeatmapsResponse)
async def all_available_heatmaps(
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_service: HeatmapsService = Depends(get_heatmaps_service),  # noqa: B008
) -> AllHeatmapsResponse:
    return await heatmaps_service.all_available_heatmaps(constants=constants)


@router.post("/all-heatmaps", response_model=HeatmapResponse)
async def get_all_heatmaps(
    request: HeatmapRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapResponse:
    responses = []

    for type in constants.AVAILABLE_HEATMAPS:
        responses.append(
            await heatmaps_facede_service.generate_heatmap(
                request=request,
                type=type,
            )
        )

    return HeatmapResponse(
        success=True,
    )


@router.get("/all-heatmaps-type", response_model=HeatmapResponse)
async def get_all_heatmaps_type(
    model: str,
    dataset: str,
    label: str,
    type: str,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
    container=Depends(get_container),  # noqa: B008
) -> HeatmapResponse:
    filenames: list[str] = os.listdir(os.path.join(container.datasets_path, dataset, label))

    request = HeatmapRequest(
        model=model,
        dataset=dataset,
        overwrite=False,
        images=[LabelFilenameTarget(label=label, filename=filename, target=None) for filename in filenames],
    )

    return await heatmaps_facede_service.generate_heatmap(
        request=request,
        type=type,
    )


@router.post("/unsupervised-heatmaps", response_model=UnsupervisedHeatmapsResponse)
async def unsupervised_heatmaps(
    request: UnsupervisedHeatmapsRequest,
    heatmaps_service: HeatmapsService = Depends(get_heatmaps_service),  # noqa: B008
) -> UnsupervisedHeatmapsResponse:
    return await heatmaps_service.unsupervised_heatmaps(request=request)


@router.post("/heatmaps-statistics-image", response_model=HeatmapStatisticResponse)
async def heatmap_statistics(
    request: HeatmapStatisticRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapStatisticResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics(request=request)


@router.post("/heatmaps-statistics-dataset", response_model=HeatmapStatisticsDatasetResponse)
async def heatmap_statistics_dataset(
    request: HeatmapStatisticsDatasetRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapStatisticsDatasetResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_dataset(request=request)


@router.post("/heatmaps-statistics-dataset-many", response_model=HeatmapStatisticsDatasetManyResponse)
async def heatmap_statistics_dataset_many(
    request: HeatmapStatisticsDatasetManyRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapStatisticsDatasetManyResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_dataset_many(request=request)


@router.post("/heatmaps-statistics-dataset-shapes", response_model=ShapeMetricsDatasetResponse)
async def heatmap_statistics_dataset_shapes(
    request: ShapeMetricsDatasetRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> ShapeMetricsDatasetResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_dataset_shapes(request=request)


@router.post("/heatmaps-statistics-label", response_model=HeatmapStatisticsLabelResponse)
async def heatmap_statistics_label(
    request: HeatmapStatisticsLabelRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapStatisticsLabelResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_label(request=request)


@router.post("/heatmaps-statistics-label-many", response_model=HeatmapStatisticsLabelManyResponse)
async def heatmap_statistics_label_many(
    request: HeatmapStatisticsLabelManyRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> HeatmapStatisticsLabelManyResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_label_many(request=request)


@router.post("/heatmaps-statistics-label-shapes", response_model=ShapeMetricsLabelResponse)
async def heatmap_statistics_label_shapes(
    request: ShapeMetricsLabelRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
) -> ShapeMetricsLabelResponse:
    return await heatmaps_facede_service.generate_heatmap_statistics_label_shapes(request=request)
