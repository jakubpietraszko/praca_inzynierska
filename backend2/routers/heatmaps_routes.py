from fastapi import APIRouter, Depends

router = APIRouter()

@router.post('/activations')
async def activations():
    raise NotImplemented