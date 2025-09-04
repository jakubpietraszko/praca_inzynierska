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


@router.post('/all-heatmaps', response_model=GradientResponse)
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
    responses = list()

    responses.append(gradient_service.gradient(request=request,
                                               container=container,
                                               models_service=models_service,
                                               datasets_service=datasets_service,
                                               utilities_service=utilities_service,
                                               constants=constants))
    responses.append(gradient_service.gradient_squared(request=request,
                                                       container=container,
                                                       models_service=models_service,
                                                       datasets_service=datasets_service,
                                                       utilities_service=utilities_service,
                                                       constants=constants))
    responses.append(gradient_service.deconvolution(request=request,
                                                    container=container,
                                                    models_service=models_service,
                                                    datasets_service=datasets_service,
                                                    utilities_service=utilities_service,
                                                    constants=constants))
    responses.append(gradient_service.guided_backpropagation(request=request,
                                                             container=container,
                                                             models_service=models_service,
                                                             datasets_service=datasets_service,
                                                             utilities_service=utilities_service,
                                                             constants=constants))
    responses.append(gradient_service.input_gradient(request=request,
                                                     container=container,
                                                     models_service=models_service,
                                                     datasets_service=datasets_service,
                                                     utilities_service=utilities_service,
                                                     constants=constants))
    responses.append(gradient_noise_service.smooth_grad(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))
    responses.append(gradient_noise_service.smooth_grad_squared(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(gradient_noise_service.var_grad(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(integrated_gradients_service.integrated_gradient(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    ))

    responses.append(cam_service.grad_cam(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(cam_service.grad_cam_plus_plus(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(cam_service.group_cam(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(cam_service.hires_cam(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(cam_service.wgrad_cam(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(cam_service.ablation_cam(
        request=request,
        container=container,
        models_service=models_service,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants
    ))

    responses.append(probing_image_service.xrai(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    ))

    responses.append(probing_image_service.rise(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    )
    )

    responses.append(probing_image_service.exact_shap(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    ))

    responses.append(probing_image_service.kernel_shap(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    ))

    responses.append(probing_image_service.lime(
        request=request,
        container=container,
        datasets_service=datasets_service,
        utilities_service=utilities_service,
        constants=constants,
        models_service=models_service,
    ))

    return GradientResponse(
        success=True,
    )
