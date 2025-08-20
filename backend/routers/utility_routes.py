import os
import base64
from fastapi import APIRouter, Depends
from services.container import get_dataset_service
from services.dataset_service import DatasetService
from interfaces.fetch_image import FetchImageRequest, FetchImageResponse
from interfaces.env import EnvResponse

router = APIRouter()


@router.get('/env', response_model=EnvResponse)
async def get_env():
    env = os.getenv('APP_ENV', 'development')
    return EnvResponse(env=env)


@router.post('/fetch-image', response_model=FetchImageResponse)
async def fetch_image(
    request: FetchImageRequest,
    dataset_service: DatasetService = Depends(get_dataset_service)
):
    try:
        path = os.path.join(
            dataset_service.datasets_path, 
            request.dataset, 
            request.label, 
            request.filename
        )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        with open(path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')
            return FetchImageResponse(image=image)
            
    except Exception as e:
        print(f"Error fetching image: {e}")
        return FetchImageResponse(image="")
