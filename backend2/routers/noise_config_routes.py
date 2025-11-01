from fastapi import APIRouter, Depends

from interfaces.noise_config import NoiseConfig
from services.container import get_noise_config_service

router = APIRouter()


@router.get("/", status_code=200)
@router.get("", status_code=200)
async def get_noise_config(noise_config_service=Depends(get_noise_config_service)) -> NoiseConfig:  # noqa: B008
    return await noise_config_service.noise_config


@router.put("/", status_code=200)
@router.put("", status_code=200)
async def update_noise_config(
    noise_config: NoiseConfig,
    noise_config_service=Depends(get_noise_config_service),  # noqa: B008
) -> dict:
    await noise_config_service.save_noise_config(noise_config)
    return {"success": True}
