import os

from fastapi import APIRouter, Depends
from interfaces.gradient import GradientRequest, GradientResponse
from constans import Constants
from services.container import get_constants

router = APIRouter()

@router.get('/all-heatmaps', response_model=GradientResponse)
async def get_all_heatmaps(request: GradientRequest,
                          constants: Constants = Depends(
                              get_constants
                          )
                           ):
    return {"message": "List of all heatmaps"}