import os

from fastapi import APIRouter, Depends
from interfaces.gradient import GradientRequest, GradientResponse
from constans import Constants
from services.container import get_constants

from services.container import ServiceContainer, get_container
from services.container import ModelsService, get_models_service
from services.container import HeatmapsService, get_heatmaps_service
from services.container import UtilitiesService, get_utilities_service
from services.container import ModelsDatasetsService, get_models_datasets_service
from services.container import DatasetsService, get_datasets_service
from services.container import GradientService, get_gradient_service
from services.container import CAMService, get_cam_service
from services.container import IntegratedGradientsService, get_integrated_gradients_service
from services.container import ProbingImageService, get_probing_image_service
from services.container import GradientNoiseService, get_gradient_noise_service


router = APIRouter()


@router.get('/all-heatmaps', response_model=GradientResponse)
async def get_all_heatmaps(request: GradientRequest,
                           container: ServiceContainer = Depends(
                               get_container),
                           models_service: ModelsService = Depends(
                               get_models_service),
                           heatmaps_service: HeatmapsService = Depends(
                               get_heatmaps_service),
                           utilities_service: UtilitiesService = Depends(
                               get_utilities_service),
                           models_datasets_service: ModelsDatasetsService = Depends(
                               get_models_datasets_service),
                           datasets_service: DatasetsService = Depends(
                               get_datasets_service),
                           constants: Constants = Depends(get_constants),
                           gradient_service: GradientService = Depends(
                               get_gradient_service),
                           cam_service: CAMService = Depends(
                               get_cam_service),
                           integrated_gradients_service: IntegratedGradientsService = Depends(
                               get_integrated_gradients_service),
                           probing_image_service: ProbingImageService = Depends(
                               get_probing_image_service),
                           gradient_noise_service: GradientNoiseService = Depends(
                               get_gradient_noise_service),

                           ):
    

    return {"message": "List of all heatmaps"}
