from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(manifest_path)


def _image_vector(image_path: str | Path, image_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(image_size)
    image_array = np.asarray(image)
    grayscale = np.asarray(image.convert("L"))
    hog_features = hog(
        grayscale,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    hist_parts = []
    for channel in range(3):
        histogram, _ = np.histogram(image_array[:, :, channel], bins=16, range=(0, 255), density=True)
        hist_parts.append(histogram)
    return np.concatenate([hog_features, *hist_parts]).astype(np.float32)


def build_feature_matrix(frame_paths: Iterable[str | Path]) -> np.ndarray:
    vectors = [_image_vector(path) for path in frame_paths]
    return np.vstack(vectors)
