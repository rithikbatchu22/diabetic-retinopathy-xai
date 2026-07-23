from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from src.data.preprocessing import crop_black_borders


@dataclass
class QualityResult:
    blur_ok: bool
    brightness_ok: bool
    quality_ok: bool
    blur_score: float
    mean_intensity: float
    message: str


def check_image_quality(
    img_rgb_uint8: np.ndarray,
    blur_threshold: float = 15.0,         # << updated (was 100)
    brightness_low: float = 30.0,
    brightness_high: float = 220.0,
    crop_black: bool = True,
    resize_for_check: int = 384,
) -> QualityResult:
    """
    Simple image quality checks for fundus images.

    Why these checks:
    - Blur (Laplacian variance): detects out-of-focus images
    - Brightness mean: detects under/over-exposed images

    Why crop/resize:
    - Fundus images include black borders and vary in resolution.
      Cropping + resizing stabilizes the blur score so one threshold is usable.
    """
    if img_rgb_uint8 is None:
        raise ValueError("img_rgb_uint8 is None")
    if img_rgb_uint8.dtype != np.uint8:
        raise ValueError("img_rgb_uint8 must be uint8")
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError("img_rgb_uint8 must have shape (H, W, 3)")

    img = img_rgb_uint8

    if crop_black:
        img = crop_black_borders(img)

    if resize_for_check is not None:
        img = cv2.resize(img, (resize_for_check, resize_for_check), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_ok = blur_score >= blur_threshold

    mean_intensity = float(gray.mean())
    brightness_ok = brightness_low <= mean_intensity <= brightness_high

    quality_ok = blur_ok and brightness_ok

    problems = []
    if not blur_ok:
        problems.append(f"Too blurry (blur score {blur_score:.1f} < {blur_threshold})")
    if not brightness_ok:
        problems.append(f"Bad exposure (mean intensity {mean_intensity:.1f})")

    message = "OK" if quality_ok else " ; ".join(problems)

    return QualityResult(
        blur_ok=blur_ok,
        brightness_ok=brightness_ok,
        quality_ok=quality_ok,
        blur_score=blur_score,
        mean_intensity=mean_intensity,
        message=message,
    )