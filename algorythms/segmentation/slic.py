import cv2
import numpy as np

def slic(
        image: np.ndarray,
        numberOfSuperpixels: int = 50,
        iterations: int = 10,
    ):
    assert image.ndim == 3

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image,
        region_size=numberOfSuperpixels,
    )
    slic.iterate(iterations)
    labels = slic.getLabels()
    masks = slic.getLabelContourMask()
    return labels, masks