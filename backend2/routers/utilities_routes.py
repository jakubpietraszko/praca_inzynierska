from fastapi import APIRouter, Depends
from services.container import get_utilities_service

from interfaces.env import EnvResponse

router = APIRouter()

@router.get('/env', response_model=EnvResponse)
async def get_env(utilities_service=Depends(get_utilities_service)):
    env = utilities_service.env()
    return EnvResponse(env=env)