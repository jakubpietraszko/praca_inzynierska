import numpy as np
from fastapi import APIRouter, Depends
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.unsupervised_activations import UnsupervisedActivationsRequest, UnsupervisedActivationsResponse
from services.container import ActivationsService
from services.container import get_activations_service


router = APIRouter()


@router.post('/activations', response_model=ActivationsResponse)
async def activations(request: ActivationsRequest,
                      activations_service: ActivationsService = Depends(
                          get_activations_service
                      )
                      ):
    response = await activations_service.activations(
        request=request,
    )
    return response

@router.post('/unsupervised-activations', response_model=UnsupervisedActivationsResponse)
async def unsupervised_activations(request: UnsupervisedActivationsRequest,
                      activations_service: ActivationsService = Depends(
                          get_activations_service
                      )
                      ):
    response = await activations_service.unsupervised_activations(
        request=request,
    )
    return response
