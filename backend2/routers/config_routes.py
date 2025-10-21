from fastapi import APIRouter, Depends

from services.config_service import ConfigModel
from services.container import get_config_service

router = APIRouter()


@router.get("/", status_code=200)
@router.get("", status_code=200)
async def get_config(config_service=Depends(get_config_service)) -> ConfigModel:  # noqa: B008
    return await config_service.config


@router.put("/", status_code=200)
@router.put("", status_code=200)
async def update_config(config: ConfigModel, config_service=Depends(get_config_service)) -> dict:  # noqa: B008
    await config_service.save_config(config)
    return {"success": True}
