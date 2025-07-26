import cv2
import numpy as np

def seed(image: np.ndarray,
         numberOfSupePixels: int = 50,
         levels: int = 2,
         iterations: int = 10
         ):
    assert image.ndim == 3, "Image must be a 3D array (H, W, C)"    

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image_width=image.shape[1],
        image_height=image.shape[0],
        image_channels=image.shape[2],
        num_superpixels=numberOfSupePixels,
        num_levels=levels,
    )
    seeds.iterate(image, iterations)
    labels = seeds.getLabels()
    masks = seeds.getLabelContourMask()
    return labels, masks


