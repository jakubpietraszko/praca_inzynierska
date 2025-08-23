import numpy as np
import cv2
from typing import Tuple

class HeatmapsService:
    def __init__(self,
                 dataset_path: str,
                 models_datasets_path: str):
        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path

    def activations(self) -> Tuple[np.ndarray, int]:
        pass