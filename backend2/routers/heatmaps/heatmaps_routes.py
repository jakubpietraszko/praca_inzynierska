import os

from fastapi import APIRouter, Depends

from constans import Constants
from interfaces.all_heatmaps import AllHeatmapsResponse
from interfaces.gradient import (
    GradientRequest,
    GradientResponse,
    GradientStatisticRequest,
    GradientStatisticResponse,
    GradientStatisticsDatasetManyRequest,
    GradientStatisticsDatasetManyResponse,
    GradientStatisticsDatasetRequest,
    GradientStatisticsDatasetResponse,
    GradientStatisticsLabelManyRequest,
    GradientStatisticsLabelManyResponse,
    GradientStatisticsLabelRequest,
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
):
    response = await heatmaps_service.all_available_heatmaps(constants=constants)
    return response


@router.post("/all-heatmaps", response_model=GradientResponse)
async def get_all_heatmaps(
    request: GradientRequest,
    constants: Constants = Depends(get_constants),  # noqa: B008
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    responses = []

    for type in constants.AVAILABLE_HEATMAPS:
        responses.append(
            await heatmaps_facede_service.generate_heatmap(
                request=request,
                type=type,
            )
        )

    return GradientResponse(
        success=True,
    )


@router.get("/all-heatmaps-type", response_model=GradientResponse)
async def get_all_heatmaps_type(
    model: str,
    dataset: str,
    label: str,
    type: str,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
    container=Depends(get_container),  # noqa: B008
):
    filenames = os.listdir(os.path.join(container.datasets_path, dataset, label))

    request = GradientRequest(
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
):
    response = await heatmaps_service.unsupervised_heatmaps(request=request)
    return response


@router.post("/heatmaps-statistics-image", response_model=GradientStatisticResponse)
async def heatmap_statistics(
    request: GradientStatisticRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics(request=request)
    return response


@router.post("/heatmaps-statistics-dataset", response_model=GradientStatisticsDatasetResponse)
async def heatmap_statistics_dataset(
    request: GradientStatisticsDatasetRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_dataset(request=request)
    return response


@router.post("/heatmaps-statistics-dataset-many", response_model=GradientStatisticsDatasetManyResponse)
async def heatmap_statistics_dataset_many(
    request: GradientStatisticsDatasetManyRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_dataset_many(request=request)
    return response


@router.post("/heatmaps-statistics-dataset-shapes", response_model=ShapeMetricsDatasetResponse)
async def heatmap_statistics_dataset_shapes(
    request: ShapeMetricsDatasetRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_dataset_shapes(request=request)
    return response


@router.post("/heatmaps-statistics-label", response_model=GradientStatisticsDatasetResponse)
async def heatmap_statistics_label(
    request: GradientStatisticsLabelRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_label(request=request)
    return response


@router.post("/heatmaps-statistics-label-many", response_model=GradientStatisticsLabelManyResponse)
async def heatmap_statistics_label_many(
    request: GradientStatisticsLabelManyRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_label_many(request=request)
    return response


@router.post("/heatmaps-statistics-label-shapes", response_model=ShapeMetricsLabelResponse)
async def heatmap_statistics_label_shapes(
    request: ShapeMetricsLabelRequest,
    heatmaps_facede_service: HeatmapsFacedeService = Depends(get_heatmaps_facede_service),  # noqa: B008
):
    response = await heatmaps_facede_service.generate_heatmap_statistics_label_shapes(request=request)
    return response
