from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import PreprocessConfig, preprocess_image
from src.data.augmentation import AugmentConfig, get_train_transforms, get_val_transforms


class DRDataset(Dataset):
    """
    Custom PyTorch Dataset for Diabetic Retinopathy.
    
    Why:
    We need to load images on-the-fly to avoid loading the entire 9.6GB dataset into RAM.
    """
    
    def __init__(
        self,
        split_csv: str,
        images_dir: str,
        mode: str = "train",  # "train" or "val"
        preprocess_cfg: Optional[PreprocessConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
    ):
        super().__init__()
        
        self.images_dir = Path(images_dir)
        self.mode = mode
        
        # Load the CSV (contains id_code and diagnosis)
        self.df = pd.read_csv(split_csv)
        
        # Initialize configs
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()
        self.augment_cfg = augment_cfg or AugmentConfig()
        
        # Select transforms based on mode
        if self.mode == "train":
            self.transforms = get_train_transforms(self.augment_cfg)
        else:
            self.transforms = get_val_transforms(self.augment_cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load one image, preprocess, augment, and return tensor + label.
        """
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = int(row["diagnosis"])
        
        # Construct image path
        img_path = self.images_dir / f"{img_id}.png"
        
        # 1. Preprocess (crop, resize) -> returns uint8 RGB numpy array
        img_rgb = preprocess_image(img_path, self.preprocess_cfg)
        
        # 2. Apply Albumentations transforms (augment + normalize + to tensor)
        # Albumentations expects image in RGB format
        transformed = self.transforms(image=img_rgb)
        img_tensor = transformed["image"]
        
        return img_tensor, label