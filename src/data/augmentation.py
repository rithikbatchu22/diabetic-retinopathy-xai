from __future__ import annotations

from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class AugmentConfig:
    """
    Configuration for image transforms.

    Why:
    Keeping transform settings in one place makes experiments reproducible.
    """
    image_size: int = 384


def get_train_transforms(cfg: AugmentConfig) -> A.Compose:
    """
    Augmentations for training only.

    Why:
    DR datasets are relatively small + imbalanced. Augmentation helps:
    - reduce overfitting
    - improve robustness to capture variations (rotation, lighting, etc.)
    """
    return A.Compose([
        # Geometric variations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.08,
            rotate_limit=15,
            border_mode=0,
            p=0.7,
        ),

        # Lighting / color variations
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.3),

        # Normalize for ImageNet-pretrained models (EfficientNet, ResNet, etc.)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        # Convert to PyTorch tensor in CHW format
        ToTensorV2(),
    ])


def get_val_transforms(cfg: AugmentConfig) -> A.Compose:
    """
    Validation/Test transforms (no augmentation).

    Why:
    We want evaluation to reflect real performance, not random augmentations.
    """
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])