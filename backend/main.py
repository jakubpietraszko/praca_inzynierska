from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DIR = '/app/DIR'

@app.get("/")
def read_root():
    return {"message": "backend"}

@app.get("/dir")
def read_dir():
    try:
        files = os.listdir(DIR)
        return {"files": files}
    except FileNotFoundError:
        return {"error": "Directory not found"}
    
@app.get("/models")
def read_models():
    try:
        models = os.listdir(os.path.join(DIR, 'models'))
        return {"models": models}
    except FileNotFoundError:
        return {"error": "Models directory not found"}
    
@app.get("/datasets")
def read_dataset():
    try:
        dataset = os.listdir(os.path.join(DIR, 'datasets'))
        return {"dataset": dataset}
    except FileNotFoundError:
        return {"error": "Dataset directory not found"}

@app.get("/slic")
def slic():
    raise NotImplemented

@app.get("/seed")
def seed():
    raise NotImplemented

@app.get("/mask")
def mask():
    raise NotImplemented

@app.get("/grad")
def grad():
    raise NotImplemented

@app.get("/integrated-grad")
def integrated_grad():
    raise NotImplemented

@app.get("/smooth-grad")
def smooth_grad():
    raise NotImplemented

@app.get("/grad-cam")
def grad_cam():
    raise NotImplemented

@app.get("/grad-cam-plus-plus")
def grad_cam_plus_plus():
    raise NotImplemented



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)