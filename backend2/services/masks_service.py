import numpy as np
import cv2
from typing import Tuple


class MasksService:
    def __init__(self,
                 dataset_path: str,
                 models_datasets_path: str):
        self.dataset_path = dataset_path
        self.models_datasets_path = models_datasets_path

    def rise(self,
             height: int,
             width: int,
             scale: int = 10,
             number_of_random_mask: int = 25,
             threshold: float = 0.5,
             ) -> np.ndarray:
        masks = np.random.random((number_of_random_mask, height // scale, width // scale))
        masks = np.where(masks > threshold, 1, 0).astype(np.uint8)

        result = np.zeros((number_of_random_mask, height, width), dtype=np.uint8)

        for i in range(len(masks)):
            result[i] = cv2.resize(
                masks[i],
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

        return result

    def seed(self,
             image_np: np.ndarray,
             height: int,
             width: int,
             dim: int,
             super_pixels: int = 20
             ) -> Tuple[np.ndarray, int]:
        LEVELS = 2
        ITERATIONS = 5
        
        if len(image_np.shape) == 3:
            resized_image = cv2.resize(image_np, (width, height))
        else:
            resized_image = cv2.resize(image_np, (width, height))
        
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width=width,
            image_height=height,
            image_channels=dim,
            num_superpixels=super_pixels,
            num_levels=LEVELS
        )

        seeds.iterate(resized_image, ITERATIONS)
        labels = seeds.getLabels()
        return labels, np.unique(labels).shape[0]
    
    def slic(self,
             image_np: np.ndarray,
             height: int,
             width: int,
             dim: int,
             super_pixels: int = 20
             ) -> Tuple[np.ndarray, int]:
        region_size = np.sqrt((height * width) / super_pixels)
        region_size = int(region_size)
        ITERATIONS = 5
        
        if len(image_np.shape) == 3:
            resized_image = cv2.resize(image_np, (width, height))
        else:
            resized_image = cv2.resize(image_np, (width, height))
        
        slic = cv2.ximgproc.createSuperpixelSLIC(
            resized_image,
            region_size=region_size,

        )
        slic.iterate(ITERATIONS)
        labels = slic.getLabels()
        return labels, np.unique(labels).shape[0]
