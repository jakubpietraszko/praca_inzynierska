from interfaces.gradient import GradientRequest, GradientResponse


class HeatmapsFacedeService:

    def __init__(self):
        from services.container import GradientService, get_gradient_service
        from services.container import IntegratedGradientsService, get_integrated_gradients_service
        from services.container import GradientNoiseService, get_gradient_noise_service
        from services.container import CAMService, get_cam_service

        from services.container import ServiceContainer, get_container
        from services.container import DatasetsService, get_datasets_service
        from services.container import UtilitiesService, get_utilities_service
        from services.container import Constants, get_constants
        from services.container import ModelsService, get_models_service
        from services.container import LRPService, get_lrp_service
        from services.container import ProbingImageService, get_probing_image_service

        self.get_gradient_service = get_gradient_service
        self.get_integrated_gradients_service = get_integrated_gradients_service
        self.get_gradient_noise_service = get_gradient_noise_service
        self.get_cam_service = get_cam_service
        self.get_lrp_service = get_lrp_service
        self.get_probing_image_service = get_probing_image_service

        self.get_container = get_container
        self.get_datasets_service = get_datasets_service
        self.get_utilities_service = get_utilities_service
        self.get_constants = get_constants
        self.get_models_service = get_models_service

    async def generate_heatmap(
            self,
            request: GradientRequest,
            type: str,
    ) -> GradientResponse:

        args = {
            'request': request,
            'container': self.get_container(),
            'datasets_service': self.get_datasets_service(),
            'utilities_service': self.get_utilities_service(),
            'constants': self.get_constants(),
            'models_service': self.get_models_service()
        }

        if type == self.get_constants().GRADIENT:
            return await self.get_gradient_service().gradient(**args)
        elif type == self.get_constants().GRADIENT_SQUARED:
            return await self.get_gradient_service().gradient_squared(**args)
        elif type == self.get_constants().DECONVOLUTION:
            return await self.get_gradient_service().deconvolution(**args)
        elif type == self.get_constants().GUIDED_BACKPROPAGATION:
            return await self.get_gradient_service().guided_backpropagation(**args)
        elif type == self.get_constants().INPUT_GRADIENT:
            return await self.get_gradient_service().input_gradient(**args)
        elif type == self.get_constants().INTEGRATED_GRADIENT:
            return await self.get_integrated_gradients_service().integrated_gradient(**args)
        elif type == self.get_constants().SMOOTH_GRADIENT:
            return await self.get_gradient_noise_service().smooth_grad(**args)
        elif type == self.get_constants().SMOOTH_GRADIENT_SQUARE:
            return await self.get_gradient_noise_service().smooth_grad_squared(**args)
        elif type == self.get_constants().VAR_GRADIENT:
            return await self.get_gradient_noise_service().var_grad(**args)

        elif type == self.get_constants().GRAD_CAM:
            return await self.get_cam_service().grad_cam(**args)
        elif type == self.get_constants().GRAD_CAM_PLUS_PLUS:
            return await self.get_cam_service().grad_cam_plus_plus(**args)
        elif type == self.get_constants().GROUP_CAM:
            return await self.get_cam_service().group_cam(**args)
        elif type == self.get_constants().HIRES_CAM:
            return await self.get_cam_service().hires_cam(**args)
        elif type == self.get_constants().WGRAD_CAM:
            return await self.get_cam_service().wgrad_cam(**args)
        elif type == self.get_constants().ABLATION_CAM:
            return await self.get_cam_service().ablation_cam(**args)
        elif type == self.get_constants().XGRAD_CAM:
            return await self.get_cam_service().xgrad_cam(**args)

        elif type == self.get_constants().LIME:
            return await self.get_probing_image_service().lime(**args)
        elif type == self.get_constants().XRAI:
            return await self.get_probing_image_service().xrai(**args)
        elif type == self.get_constants().KERNEL_SHAP:
            return await self.get_probing_image_service().kernel_shap(**args)

        return GradientResponse(
            success=True
        )
