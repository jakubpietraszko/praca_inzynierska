import numpy as np
import cv2
from typing import Tuple
from interfaces.all_heatmaps import AllHeatmapsResponse

class HeatmapsService:
    def __init__(self,
                 dataset_path: str,
                 models_datasets_path: str):
        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path

    async def activations(self) -> Tuple[np.ndarray, int]:
        pass

    async def all_available_heatmaps(self, constants):
        heatmaps = constants.AVAILABLE_HEATMAPS
        return AllHeatmapsResponse(
            heatmaps=heatmaps
        )