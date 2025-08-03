from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import importlib
from fastapi.responses import JSONResponse
import base64
from interfaces.respones.listOfStrings import ListOfStringsResponse
from interfaces.respones.error import ErrorResponse
from interfaces.requests.path import PathRequest
from interfaces.requests.all_images import AllImagesRequest
from interfaces.requests.use_model import UseModelRequest
import importlib.util

ENV = os.getenv('APP_ENV', 'development')

if ENV == 'development':
    DIR = os.path.join('..', 'DIR')
else:
    DIR = os.path.join('/app', 'DIR')

DATASETS = os.path.join(DIR, 'datasets')
MODELS = os.path.join(DIR, 'models')
MODEL = {
    'name': None,
    'model': None
}
DATASET = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/use-model')
def use_model(request: UseModelRequest):
    model_path = os.path.join(MODELS, request.model_name, 'model.py')

    spec = importlib.util.spec_from_file_location('loadModel', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print(os.curdir)

    try:
        if hasattr(module, 'loadModel'):
            # Example: pass parameters to loadModel, e.g., param1 and param2
            # Replace 'param1', 'param2' with actual parameters as needed
            MODEL['name'] = request.model_name
            MODEL['model'] = getattr(module, 'loadModel')()
            print(f"Model {request.model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {request.model_name}: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    
    return JSONResponse(content={"message": f"Model {request.model_name} loaded successfully."})

@app.get('/env')
def get_env():
    return {"env": ENV}

@app.post('/load-image')
async def post_main(request: PathRequest):
    photo_path = os.path.join(
        DATASETS, request.path
    )
    with open(photo_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return JSONResponse(content={"image": encoded_string})

@app.get('/load-model')
def load_model():
    allModelPaths = list()
    for (root, dirs, files) in os.walk(MODELS, topdown=True):
        if 'model.py' in files:
            allModelPaths.append(
                os.path.join(root, 'model.py')
            )
    return JSONResponse(content={"model_paths": allModelPaths})



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
        models = os.listdir(MODELS)
        return {"models": models}
    except FileNotFoundError:
        return {"error": "Models directory not found"}
    
@app.get("/datasets")
def read_dataset():
    try:
        dataset = os.listdir(DATASETS)
        return {"dataset": dataset}
    except FileNotFoundError:
        return {"error": "Dataset directory not found"}
    
@app.get('/classes')
def get_classes(request: PathRequest):
    path = os.path.join(DATASETS, request.path)
    try:
        listOfClasses = [dir for dir in os.listdir(path)]
        return ListOfStringsResponse(strings=listOfClasses)
    except FileNotFoundError:
        return ErrorResponse(
            error="File not found",
            code=404,
            details={"path": request.path}
        )

@app.get('/all-images')
def get_images(request: AllImagesRequest):
    path = os.path.join(DATASETS, request.dataset, request.label)
    try:
        images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        return ListOfStringsResponse(strings=images)
    except FileNotFoundError:
        return ErrorResponse(
            error="File not found",
            code=404,
            details={"dataset": request.dataset, "label": request.label}
        )





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