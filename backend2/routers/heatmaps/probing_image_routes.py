import os
import numpy as np
import cv2
import torch
from fastapi import APIRouter, Depends
from interfaces.label_filename import LabelFilename
from interfaces.activations import ActivationsRequest, ActivationsResponse
from interfaces.gradient import GradientRequest, GradientResponse
from services.container import ServiceContainer, ModelsService, HeatmapsService, UtilitiesService, ModelsDatasetsService, DatasetsService
from services.container import get_container, get_models_service, get_heatmaps_service, get_utilities_service, get_models_datasets_service, get_datasets_service, get_constants
from constans import Constants
from PIL import Image
import matplotlib.pyplot as plt


router = APIRouter()


@router.post('/xrai', response_model=GradientResponse)
async def xrai(request: GradientRequest):
    pass


@router.post('/rise', response_model=GradientResponse)
async def rise(request: GradientRequest):
    pass


@router.post('/exact-shap', response_model=GradientResponse)
async def exact_shap(request: GradientRequest):
    pass


@router.post('/kernel-shap', response_model=GradientResponse)
async def kernel_shap(request: GradientRequest):
    pass


@router.post('/lime', response_model=GradientResponse)
async def lime(request: GradientRequest):
    pass
