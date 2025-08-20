from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from routers.data_routes import router as data_router
from routers.utility_routes import router as utility_router
from routers.mask_routes import router as mask_router
from routers.gradient_routes import router as gradient_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(utility_router)
app.include_router(data_router)
app.include_router(mask_router)
app.include_router(gradient_router)


@app.get("/")
async def root():
    return {
        "message": "heat-map generation",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
