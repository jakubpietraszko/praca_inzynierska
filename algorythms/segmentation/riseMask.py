import numpy as np
import cv2

def riseMask(height: int, width: int, scale: int = 10, numberOfRandomMask: int = 25):
    THRESHOLD = 0.5
    masks = np.random.random(
        (numberOfRandomMask, height // scale, width // scale),
    )
    masks = np.where(masks > THRESHOLD, 1, 0).astype(np.uint8)

    temp = np.zeros((numberOfRandomMask, height, width), dtype=np.uint8)
    print(temp.shape)

    for i in range(len(masks)):
        temp[i] = cv2.resize(
            masks[i],
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    return temp
