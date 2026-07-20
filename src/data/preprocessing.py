from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    image_size: int = 384
    crop_black: bool = True
    enhance_contrast: bool = False  # keep False for baseline


def read_image_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """
    Read an image from disk and return RGB uint8 array: (H, W, 3).

    Why:
    - OpenCV reads as BGR by default; we convert to RGB to keep consistency
      across visualization, training, and Streamlit UI.
    """
    image_path = str(image_path)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def crop_black_borders(img_rgb: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop black/dark borders using a simple threshold on grayscale.

    Why:
    - Fundus images often have a circular field-of-view with black background.
      Cropping removes uninformative pixels and helps the model focus on retina.

    Notes:
    - tol controls what is considered "non-black".
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if not np.any(mask):
        # If the image is extremely dark, just return original
        return img_rgb

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 because slicing is exclusive
    cropped = img_rgb[y0:y1, x0:x1]
    return cropped


def ben_graham_enhancement(img_rgb: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Optional contrast enhancement commonly used in DR preprocessing.

    Why:
    - Sometimes improves visibility of lesions/vessels by normalizing illumination.

    Keep it OFF for the baseline (enhance_contrast=False).
    We'll compare later if it helps.
    """
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    enhanced = cv2.addWeighted(img_rgb, 4.0, blur, -4.0, 128)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def resize_to_square(img_rgb: np.ndarray, size: int) -> np.ndarray:
    """
    Resize image to (size, size).

    Why:
    - Neural nets expect fixed input size.
    - We use square size for EfficientNet-based models.
    """
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_image(
    image_path: Union[str, Path],
    cfg: Optional[PreprocessConfig] = None
) -> np.ndarray:
    """
    End-to-end preprocessing:
    - read RGB
    - crop black borders (optional)
    - enhancement (optional)
    - resize to cfg.image_size

    Returns:
    - RGB uint8 image of shape (image_size, image_size, 3)
    """
    cfg = cfg or PreprocessConfig()

    img = read_image_rgb(image_path)

    if cfg.crop_black:
        img = crop_black_borders(img)

    if cfg.enhance_contrast:
        img = ben_graham_enhancement(img)

    img = resize_to_square(img, cfg.image_size)
    return img


def to_float_normalized(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert uint8 RGB image to float32 [0, 1].

    Why:
    - Deep learning models train better on normalized float inputs.
    - Actual mean/std normalization for ImageNet will be applied later
      in the Dataset/Transforms step.
    """
    if img_rgb_uint8.dtype != np.uint8:
        raise ValueError("Expected uint8 image input")
    return (img_rgb_uint8.astype(np.float32) / 255.0)