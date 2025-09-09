from interfaces.gradient import GradientRequest, GradientResponse


class HeatmapsFacedeService:

    def __init__(self):
        from services.container import GradientService, get_gradient_service
        from services.container import IntegratedGradientsService, get_integrated_gradients_service
        from services.container import GradientNoiseService, get_gradient_noise_service

        self.get_gradient_service = get_gradient_service
        self.get_integrated_gradients_service = get_integrated_gradients_service
        self.get_gradient_noise_service = get_gradient_noise_service

    def generate_heatmap(
            self,
            request: GradientRequest,
            type: str,
            container,
            datasets_service,
            utilities_service,
            constants,
            models_service) -> GradientResponse:

        args = {
            'request': request,
            'container': container,
            'datasets_service': datasets_service,
            'utilities_service': utilities_service,
            'constants': constants,
            'models_service': models_service
        }

        if type == constants.GRADIENT:
            return self.get_gradient_service().gradient(**args)
        elif type == constants.GRADIENT_SQUARED:
            return self.get_gradient_service().gradient_squared(**args)
        elif type == constants.DECONVOLUTION:
            return self.get_gradient_service().deconvolution(**args)
        elif type == constants.GUIDED_BACKPROPAGATION:
            return self.get_gradient_service().guided_backpropagation(**args)
        elif type == constants.INPUT_GRADIENT:
            return self.get_gradient_service().input_gradient(**args)
        elif type == constants.INTEGRATED_GRADIENT:
            return self.get_integrated_gradients_service().integrated_gradient(**args)
        elif type == constants.SMOOTH_GRADIENT:
            return self.get_gradient_noise_service().smooth_grad(**args)
        elif type == constants.SMOOTH_GRADIENT_SQUARE:
            return self.get_gradient_noise_service().smooth_grad_squared(**args)
        elif type == constants.VAR_GRADIENT:
            return self.get_gradient_noise_service().var_grad(**args)

        return GradientResponse(
            success=True
        )
