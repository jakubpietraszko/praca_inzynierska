from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.config_service import ConfigModel, ConfigService
from services.container import get_config_service

router = APIRouter()


@router.get("/", status_code=200)
@router.get("", status_code=200)
async def get_config(config_service=Depends(get_config_service)) -> ConfigModel:
    return config_service.config


@router.put("/", status_code=200)
@router.put("", status_code=200)
async def update_config(config: ConfigModel, config_service=Depends(get_config_service)) -> dict:

    config_service.save_config(config)
    return {"success": True}
