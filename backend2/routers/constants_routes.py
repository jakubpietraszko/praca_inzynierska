from fastapi import APIRouter, Depends

from services.constants_service import ConstantsService
from services.container import get_constants_service

router = APIRouter()


@router.get("/xai-translations")
def get_xai_translations(constants_service: ConstantsService = Depends(get_constants_service)):  # noqa: B008
    return constants_service.get_translations()
