import os
import cv2
import numpy as np
from typing import List, Tuple


class MaskService:
    
    def __init__(self, output_path: str):
        self.output_path = output_path
    
    def create_directories(self, model_name: str, dataset_name: str) -> bool:
        root_path = os.path.join(self.output_path, f'{model_name}_{dataset_name}')
        
        all_dirs = ['mask_rise', 'mask_slic', 'mask_seed']
        
        try:
            os.makedirs(root_path, exist_ok=True)
            
            for dir_name in all_dirs:
                dir_path = os.path.join(root_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
                
            return True
        except Exception:
            return False
    
    def generate_rise_masks(self, 
                          shape: Tuple[int, int],
                          scale: int = 10,
                          number_of_masks: int = 25,
                          threshold: float = 0.5) -> np.ndarray:
        height, width = shape
        masks = np.random.random((number_of_masks, height // scale, width // scale))
        masks = np.where(masks > threshold, 1, 0).astype(np.uint8)

        temp = np.zeros((number_of_masks, height, width), dtype=np.uint8)

        for i in range(len(masks)):
            temp[i] = cv2.resize(
                masks[i],
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )
        
        return temp
    
    def generate_slic_masks(self, 
                          image: np.ndarray,
                          height: int,
                          width: int,
                          number_of_superpixels: int = 100,
                          iterations: int = 5) -> np.ndarray:
        region_size = int(np.sqrt((width * height) / number_of_superpixels))

        slic = cv2.ximgproc.createSuperpixelSLIC(
            image, region_size=region_size
        )

        slic.iterate(iterations)
        labels = slic.getLabels()
        return labels
    
    def generate_seed_masks(self, 
                          image: np.ndarray,
                          height: int,
                          width: int,
                          number_of_superpixels: int = 10,
                          iterations: int = 5) -> np.ndarray:
        levels = 2
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width=width,
            image_height=height,
            image_channels=image.shape[2],
            num_superpixels=number_of_superpixels,
            num_levels=levels
        )

        seeds.iterate(image, iterations)
        labels = seeds.getLabels()
        return labels
    
    def save_rise_masks(self, 
                       masks: np.ndarray, 
                       model_name: str, 
                       dataset_name: str, 
                       label: str, 
                       filename: str) -> None:
        filename_base = filename.split('.')[0]
        path = os.path.join(
            self.output_path, 
            f'{model_name}_{dataset_name}', 
            'mask_rise', 
            label, 
            filename_base
        )
        os.makedirs(path, exist_ok=True)

        for i, mask in enumerate(masks):
            mask_path = os.path.join(path, f'{i}.png')
            cv2.imwrite(mask_path, mask * 255)
    
    def save_superpixel_mask(self, 
                           mask: np.ndarray, 
                           model_name: str, 
                           dataset_name: str, 
                           label: str, 
                           filename: str, 
                           mask_type: str) -> None:
        label_path = os.path.join(
            self.output_path, 
            f'{model_name}_{dataset_name}', 
            f'mask_{mask_type}', 
            label
        )
        os.makedirs(label_path, exist_ok=True)

        mask_path = os.path.join(label_path, f'{filename.split(".")[0]}.png')
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
        cv2.imwrite(mask_path, mask)
