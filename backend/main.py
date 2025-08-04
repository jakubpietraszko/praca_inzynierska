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
from interfaces.requests.use_dataset import UseDatasetRequest
from interfaces.requests.grad import GradRequest
from interfaces.requests.all_labels import AllLabelsRequest
import importlib.util
from torchvision.datasets import ImageFolder
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time



ENV = os.getenv('APP_ENV', 'development')

if ENV == 'development':
    DIR = os.path.join('..', 'DIR')
else:
    DIR = os.path.join('/app', 'DIR')


PATH_DATASETS = os.path.join(DIR, 'datasets')
PATH_MODELS = os.path.join(DIR, 'models')
PATH_MODELS_DATASETS = os.path.join(DIR, 'model-dataset')

MODEL = {
    'name': None,
    'model': None,
    'last_conv': None,
    'transform': None,
    'idx_to_label': None
}
DATASET = {
    'name': None,
    'dataset': None
}

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
    model_path = os.path.join(PATH_MODELS, request.model_name, 'model.py')

    spec = importlib.util.spec_from_file_location('loadModel', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print(os.curdir)

    try:
        if hasattr(module, 'loadModel'):
            MODEL['name'] = request.model_name
            model, last_conv, transform, idx_to_label = getattr(module, 'loadModel')()
            MODEL['model'] = model
            MODEL['last_cnn'] = last_conv
            MODEL['transform'] = transform
            MODEL["idx_to_label"] = idx_to_label

            print(f"Model {request.model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {request.model_name}: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    
    return JSONResponse(content={"message": f"Model {request.model_name} loaded successfully."})

@app.get('/use-dataset')
async def use_dataset(request: UseDatasetRequest):
    path = os.path.join(PATH_DATASETS, request.dataset_name)
    try:
        dataset = ImageFolder(path, transform=MODEL['transform'])
        DATASET['name'] = request.dataset_name
        DATASET['dataset'] = dataset
        print(f"Dataset {request.dataset_name} loaded successfully.")
        print(f"Dataset size: {len(dataset)} images.")
        return JSONResponse(content={"message": f"Dataset {request.dataset_name} loaded successfully."})
    except FileNotFoundError:
        print(f"Dataset {request.dataset_name} not found.")
        return JSONResponse(
            content={"error": f"Dataset {request.dataset_name} not found."},
            status_code=404
        )
    except Exception as e:
        print(f"Error loading dataset {request.dataset_name}: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get('/env')
def get_env():
    return {"env": ENV}

@app.post('/load-image')
async def post_main(request: PathRequest):
    photo_path = os.path.join(
        PATH_DATASETS, request.path
    )
    with open(photo_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return JSONResponse(content={"image": encoded_string})

@app.get('/load-model')
def load_model():
    allModelPaths = list()
    for (root, dirs, files) in os.walk(PATH_MODELS, topdown=True):
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
        models = os.listdir(PATH_MODELS)
        return {"models": models}
    except FileNotFoundError:
        return {"error": "Models directory not found"}
    
@app.get("/all-datasets")
def read_dataset():
    try:
        dataset = os.listdir(PATH_DATASETS)
        return {"datasets": dataset}
    except FileNotFoundError:
        return {"error": "Dataset directory not found"}
    
@app.post('/all-labels')
def get_classes(request: AllLabelsRequest):
    path = os.path.join(PATH_DATASETS, request.dataset)
    try:
        listOfClasses = [dir for dir in os.listdir(path)]
        return ListOfStringsResponse(labels=listOfClasses)
    except FileNotFoundError:
        return ErrorResponse(
            error="File not found",
            code=404,
            details={"path": request.path}
        )

@app.get('/all-images')
def get_images(request: AllImagesRequest):
    path = os.path.join(PATH_DATASETS, request.dataset, request.label)
    try:
        images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        return ListOfStringsResponse(strings=images)
    except FileNotFoundError:
        return ErrorResponse(
            error="File not found",
            code=404,
            details={"dataset": request.dataset, "label": request.label}
        )





@app.get("/mask/slic")
def slic():
    raise NotImplemented

@app.get("/mask/seed")
def seed():
    raise NotImplemented

@app.get("/mask/rise")
def mask():
    raise NotImplemented

@app.get("/activation-maps")
def activation_maps():
    raise NotImplemented


@app.get("/heat-map/grad")
async def grad(request: GradRequest):
    model_name = request.model_name
    dataset_name = request.dataset_name
    label_name = request.label_name
    filename = request.filename
    target_label = request.target_label
    normalize = request.normalize


    path = os.path.join(PATH_DATASETS, dataset_name, label_name, filename)
    print(f"Path to image: {path}")

    # time.sleep(100)
    

    model = MODEL['model']
    model.eval()
    image =  Image.open(path).convert("RGB")
    image = MODEL['transform'](image).unsqueeze(0)
    image_cp = image.clone().detach()
    image_cp.requires_grad_()
    
    output = model(image_cp)
    y_c = output[0, target_label]
    grad = torch.autograd.grad(y_c, image_cp)[0]
    result = grad.squeeze().cpu().detach().numpy()
    result = np.maximum(result, 0)

    result *= 255.0
    result = result.astype(np.uint8)

    save_path = os.path.join(PATH_MODELS_DATASETS, 'grad.png')
    plt.imsave(save_path, result, cmap='hot')

@app.get("/heat-map/integrated-grad")
def integrated_grad():
    raise NotImplemented

@app.get("/heat-map/smooth-grad")
def smooth_grad():
    raise NotImplemented

@app.get("/heat-map/var-grad")
def var_grad():
    raise NotImplemented



@app.get("/heat-map/grad-cam")
def grad_cam():
    raise NotImplemented

@app.get("/heat-map/grad-cam-plus-plus")
def grad_cam_plus_plus():
    raise NotImplemented

@app.get("/heat-map/wgrad-cam")
def wgrad_cam():
    raise NotImplemented

@app.get("/heat-map/hires-cam")
def hires_cam():
    raise NotImplemented

@app.get("/heat-map/ablation-cam")
def ablation_cam():
    raise NotImplemented

@app.get("/heat-map/group-cam")
def group_cam():
    raise NotImplemented



@app.get("/heat-map/rise")
def rise():
    raise NotImplemented

@app.get("/heat-map/exact-shap")
def exact_shap():
    raise NotImplemented

@app.get("/heat-map/kernel-shap")
def kernel_shap():
    raise NotImplemented

@app.get("heat-map/lime")
def lime():
    raise NotImplemented



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)