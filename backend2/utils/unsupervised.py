import colorsys

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, normalize


def generate_colors(num_classes: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.8 + (i % 3) * 0.1
        value = 0.7 + (i % 2) * 0.2
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


async def apply_tsne_with_augmentation(images_flat_np: np.ndarray, min_samples: int = 31) -> np.ndarray:
    images_flat_np = images_flat_np.astype(np.float64)
    images_flat_np += np.random.normal(0, 1e-10, images_flat_np.shape)

    original_size = images_flat_np.shape[0]

    if original_size < min_samples:
        additional_needed = min_samples - original_size
        random_indices = np.random.choice(original_size, size=additional_needed, replace=True)
        images_flat_np = np.vstack([images_flat_np, images_flat_np[random_indices]])

    tsne = TSNE(n_components=2, init="random", learning_rate="auto")
    images_tsne = tsne.fit_transform(images_flat_np)

    images_tsne = normalize(images_tsne, axis=0)
    scaler = MinMaxScaler()
    images_tsne = scaler.fit_transform(images_tsne)

    images_tsne = images_tsne[:original_size]

    return images_tsne
