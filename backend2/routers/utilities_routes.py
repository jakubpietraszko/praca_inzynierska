from fastapi import APIRouter, Depends

from interfaces.env import EnvResponse
from services.container import get_utilities_service

router = APIRouter()

@router.get('/env', response_model=EnvResponse)
async def get_env():
    utilities_service=Depends(get_utilities_service)
    env = await utilities_service.env()
    return EnvResponse(env=env)
