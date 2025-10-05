from fastapi import APIRouter, Depends

router = APIRouter()

@router.get('/xai-translations')
def get_xai_translations():
    return ''