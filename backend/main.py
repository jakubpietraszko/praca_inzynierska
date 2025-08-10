from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import importlib
from fastapi.responses import JSONResponse
import base64
from interfaces.requests.grad import GradRequest
import importlib.util
from torchvision.datasets import ImageFolder
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from interfaces.all_images import AllsImagesRequest, AllImagesResponse
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_models import AllModelsResponse
from interfaces.fetch_image import FetchImageRequest, FetchImageResponse
from interfaces.use_dataset import UseDatasetRequest, UseDatasetResponse
from interfaces.use_model import UseModelRequest, UseModelResponse
from interfaces.env import EnvResponse
from interfaces.model_layers import ModelLayersRequest, ModelLayersResponse


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

@app.get('/env')
async def get_env():
    return EnvResponse(env=ENV)

@app.get('/all-models')
async def all_models():
    models = os.listdir(PATH_MODELS)
    return AllModelsResponse(models=models)

@app.get('/all-datasets')
async def all_datasets():
    datasets = os.listdir(PATH_DATASETS)
    return AllDatasetsResponse(datasets=datasets)

@app.post('/all-labels')
async def all_labels(request: AllLabelsRequest):
    path = os.path.join(PATH_DATASETS, request.dataset)
    labels = [label for label in os.listdir(path)]
    return AllLabelsResponse(labels=labels)

@app.post('/all-images')
async def all_images(request: AllsImagesRequest):
    path = os.path.join(PATH_DATASETS, request.dataset, request.label)
    images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    return AllImagesResponse(images=images)

@app.post('/fetch-image')
async def fetch_image(request: FetchImageRequest):
    path = os.path.join(PATH_DATASETS, request.dataset, request.label, request.filename)
    with open(path, 'rb') as f:
        image = base64.b64encode(f.read()).decode('utf-8')
        return FetchImageResponse(image=image)
    
@app.post('/model-layers')
async def model_layers(request: ModelLayersRequest):
    response: ModelLayersResponse = _generate_layers(request)
    return response 

def _generate_layers(request: ModelLayersRequest) -> ModelLayersResponse:
    model, shape = _load_model(request.model)

    dummy_input = torch.randn(1, *shape)
    layers = list()
    hooks = list()

    def hook_fn(name):
        def hook(module, input, output):
            layers.append((name, module))
        return hook
    
    for name, module in model.named_modules():
        if name:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()
    
    result = list()

    for i, (name, layer) in enumerate(layers):
        result.append(str(layer))

    return ModelLayersResponse(layers=result)

def _load_model(model_name: str):
    path = os.path.join(PATH_MODELS, model_name, 'model.py')
    spec = importlib.util.spec_from_file_location('loadModel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model, _, _, _, shape = getattr(module, 'loadModel')()
    return model, shape








@app.get('/use-model')
async def use_model(request: UseModelRequest):
    if _use_model(request):
        return UseModelResponse(success=True)
    return UseModelResponse(success=False)

def _use_model(request: UseModelRequest) -> bool:
    path = os.path.join(PATH_MODELS, request.model, 'model.py')
    spec = importlib.util.spec_from_file_location('loadModel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        if hasattr(module, 'loadModel'):
            MODEL['name'] = request.model
            model, last_conv, transform, idx_to_label, shape = getattr(module, 'loadModel')()
            MODEL['model'] = model
            MODEL['last_cnn'] = last_conv
            MODEL['transform'] = transform
            MODEL["idx_to_label"] = idx_to_label
            return True
    except:
        return False
    return False

@app.get('/use-dataset')
async def use_dataset(request: UseDatasetRequest):
    if _use_dataset(request):
        return UseDatasetResponse(success=True)
    return UseDatasetResponse(success=False)

def _use_dataset(request: UseDatasetRequest) -> bool:
    path = os.path.join(PATH_DATASETS, request.dataset)
    try:
        dataset = ImageFolder(path, transform=MODEL['transform'])
        DATASET['name'] = request.dataset
        DATASET['dataset'] = dataset
        return True
    except:
        return False
    return False








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