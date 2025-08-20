from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from routers.datasets_routes import router as datasets_router
from routers.utilities_routes import router as utilities_router
from routers.models_routes import router as models_router
from routers.models_datasets_routes import router as models_datasets_router
from routers.masks_routes import router as masks_router
from routers.heatmaps_routes import router as heatmaps_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(utilities_router)
app.include_router(models_router)
app.include_router(models_datasets_router)
app.include_router(masks_router, prefix="/mask")
app.include_router(heatmaps_router, prefix="/heatmaps")

@app.get("/")
async def root():
    return {
        "message": "Welcome",
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
