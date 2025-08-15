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
import cv2

from interfaces.all_images import AllsImagesRequest, AllImagesResponse
from interfaces.all_labels import AllLabelsRequest, AllLabelsResponse
from interfaces.all_datasets import AllDatasetsResponse
from interfaces.all_models import AllModelsResponse
from interfaces.fetch_image import FetchImageRequest, FetchImageResponse
from interfaces.use_dataset import UseDatasetRequest, UseDatasetResponse
from interfaces.use_model import UseModelRequest, UseModelResponse
from interfaces.env import EnvResponse
from interfaces.model_layers import ModelLayersRequest, ModelLayersResponse
from interfaces.init_dir import InitDirRequest, InitDirResponse
from interfaces.masks import MasksRequest, MasksResponse
from interfaces.activations import ActivationsRequest, ActivationsResponse

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
    model, _, _, _, shape = _load_model(request.model)

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

    model, last_conv, transform, idx_to_label, shape = getattr(module, 'loadModel')()
    return model, last_conv, transform, idx_to_label, shape

@app.post('/init-dir')
async def init_dir(request: InitDirRequest):
    success = _create_model_dataset_dirs(request)
    return InitDirResponse(success=success)

ALL_DIRS = [
    'mask_rise',
    'mask_slic',
    'mask_seed',
]

def _create_model_dataset_dirs(request: InitDirRequest):
    root_path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}')
    try:
        os.makedirs(root_path, exist_ok=True)

        for dir_name in ALL_DIRS:
            dir_path = os.path.join(root_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
    except:
        return False
    return True

@app.post('/mask/rise')
async def mask_rise(request: MasksRequest):
    model, _, _, _, shape = _load_model(request.model)
    shape = shape[1:]

    for image in request.images:
        label, filename = image.label, image.filename
        filename = filename.split('.')[0]

        path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}', 'mask_rise', label, filename)
        os.makedirs(path, exist_ok=True)

        masks = _generate_rise_masks(shape, scale=5, numberOfRandomMask=request.count)

        for i, mask in enumerate(masks):
            mask_path = os.path.join(path, f'{i}.png')
            cv2.imwrite(mask_path, mask * 255)

    return MasksResponse(success=True)

def _generate_rise_masks(shape: tuple,
                         scale: int = 10,
                         numberOfRandomMask: int = 25,
                         threshold: float = 0.5) -> np.ndarray:
    height, width = shape
    masks = np.random.random((numberOfRandomMask, height // scale, width // scale))
    masks = np.where(masks > threshold, 1, 0).astype(np.uint8)

    temp = np.zeros((numberOfRandomMask, height, width), dtype=np.uint8)

    for i in range(len(masks)):
        temp[i] = cv2.resize(
            masks[i],
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
    
    return temp

@app.post('/mask/slic')
async def mask_slic(request: MasksRequest):
    model, _, _, _, shape = _load_model(request.model)
    dim, height, width = shape
    shape = shape[1:]

    path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}', 'mask_slic')
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label, filename = image.label, image.filename
        label_path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}', 'mask_slic', label)
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(PATH_DATASETS, request.dataset, label, filename)

        image_np = cv2.imread(image_path)
        mask = _generate_slic_masks(image_np, height, width, numberOfSuperpixels=request.count)

        mask_path = os.path.join(label_path, f'{filename.split(".")[0]}.png')
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
        cv2.imwrite(mask_path, mask)

    return MasksResponse(success=True)

def _generate_slic_masks(image: np.ndarray,
                         height: int,
                         width: int,
                         numberOfSuperpixels: int = 100,
                         iterations: int = 5) -> np.ndarray:

    regionSize = int(np.sqrt((width * height) / numberOfSuperpixels))

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image, region_size=regionSize
    )

    slic.iterate(iterations)
    labels = slic.getLabels()
    print(f'max label: {np.max(labels)}')
    return labels



@app.post('/mask/seed')
async def mask_seed(request: MasksRequest):
    print('aa')
    model, _, _, _, shape = _load_model(request.model)
    dim, height, width = shape
    shape = shape[1:]

    path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}', 'mask_seed')
    os.makedirs(path, exist_ok=True)

    for image in request.images:
        label, filename = image.label, image.filename
        label_path = os.path.join(PATH_MODELS_DATASETS, f'{request.model}_{request.dataset}', 'mask_seed', label)
        os.makedirs(label_path, exist_ok=True)

        image_path = os.path.join(PATH_DATASETS, request.dataset, label, filename)

        image_np = cv2.imread(image_path)
        mask = _generate_seed_masks(image_np, height, width, numberOfSuperpixels=request.count)

        mask_path = os.path.join(label_path, f'{filename.split(".")[0]}.png')
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
        cv2.imwrite(mask_path, mask)

    return MasksResponse(success=True)

def _generate_seed_masks(image: np.ndarray,
                         height: int,
                         width: int,
                         numberOfSuperpixels: int = 10,
                         iterations: int = 5) -> np.ndarray:
    LEVELS = 2
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image_width=width,
        image_height=height,
        image_channels=image.shape[2],
        num_superpixels=numberOfSuperpixels,
        num_levels=LEVELS
    )

    seeds.iterate(image, iterations)
    labels = seeds.getLabels()
    print(f'max label: {np.max(labels)}')
    return labels



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








@app.get("/activation-maps")
async def activation_maps(request: ActivationsRequest):
    model_name = request.model
    dataset = request.dataset
    images = request.images

    model, last_conv, transform, _, _ = _load_model(model_name)

    for image in images:
        label = image.label
        filename = image.filename

        path = os.path.join(PATH_DATASETS, dataset, label, filename)
        base = filename.split('.')[0]
        print(f"Path to image: {path}")
        
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0)

        def activation_hook(module, input, output):
            activation = output.detach().cpu().numpy()
            print(f"Activation shape: {activation.shape}")

        last_conv.register_forward_hook(activation_hook)

        model.eval()
        with torch.no_grad():
            model(image)







    


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