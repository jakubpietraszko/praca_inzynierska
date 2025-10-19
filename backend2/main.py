from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.config_routes import router as config_router
from routers.constants_routes import router as constants_router
from routers.datasets_routes import router as datasets_router
from routers.edge_routes import router as edge_router
from routers.heatmaps.activations_routes import router as activations_router
from routers.heatmaps.cam_routes import router as cam_router
from routers.heatmaps.gradient_noise_routes import router as gradient_noise_router
from routers.heatmaps.gradient_routers import router as gradient_router
from routers.heatmaps.heatmaps_routes import router as heatmaps_router
from routers.heatmaps.integrated_gradient_routes import router as integrated_gradient_router
from routers.heatmaps.lrp_routes import router as lrp_router
from routers.heatmaps.probing_routes import router as probing_router
from routers.masks_routes import router as masks_router
from routers.models_datasets_routes import router as models_datasets_router
from routers.models_routes import router as models_router
from routers.statistics_routes import router as statistics_router
from routers.unsupervised_routes import router as unsupervised_router
from routers.utilities_routes import router as utilities_router
from routers.websocket_routes import router as websocket_router

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=10)
app.state.executor = executor

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(models_router)
app.include_router(models_datasets_router)
app.include_router(utilities_router)

app.include_router(masks_router, prefix="/mask")

app.include_router(edge_router, prefix="/heatmap")
app.include_router(heatmaps_router, prefix="/heatmap")
app.include_router(activations_router, prefix="/heatmap")
app.include_router(cam_router, prefix="/heatmap")
app.include_router(gradient_noise_router, prefix="/heatmap")
app.include_router(gradient_router, prefix="/heatmap")
app.include_router(probing_router, prefix='/heatmap')
app.include_router(integrated_gradient_router, prefix="/heatmap")
app.include_router(lrp_router, prefix="/heatmap")

app.include_router(unsupervised_router, prefix="/unsupervised")
app.include_router(statistics_router, prefix='/statistics')

app.include_router(websocket_router)

app.include_router(config_router, prefix="/config")
app.include_router(constants_router, prefix='/constants')

@app.get("/")
async def root():
    return {
        "message": "Welcome",
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
