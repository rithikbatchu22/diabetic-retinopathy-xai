from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.augmentation import AugmentConfig
from src.data.dataset import DRDataset
from src.data.preprocessing import PreprocessConfig
from src.models.model import create_dr_model
from src.training.trainer import TrainConfig, fit
from src.utils.seed import seed_everything


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_class_weights(train_csv: str, num_classes: int, device: str) -> torch.Tensor:
    """
    Why:
    APTOS is imbalanced (many class 0, fewer class 3/4).
    Class weights reduce bias toward majority class.
    """
    df = pd.read_csv(train_csv)
    counts = df["diagnosis"].value_counts().to_dict()

    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(1.0 / cnt)

    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()  # normalize (mean weight = 1)
    return w.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed_everything(cfg["project"]["seed"])

    image_size = int(cfg["data"]["image_size"])
    num_classes = int(cfg["data"]["num_classes"])

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    images_dir = cfg["data"]["images_dir"]

    model_name = cfg["train"]["model_name"]
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    num_workers = int(cfg["train"]["num_workers"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    preprocess_cfg = PreprocessConfig(image_size=image_size, crop_black=True, enhance_contrast=False)
    augment_cfg = AugmentConfig(image_size=image_size)

    train_ds = DRDataset(
        split_csv=train_csv,
        images_dir=images_dir,
        mode="train",
        preprocess_cfg=preprocess_cfg,
        augment_cfg=augment_cfg,
    )
    val_ds = DRDataset(
        split_csv=val_csv,
        images_dir=images_dir,
        mode="val",
        preprocess_cfg=preprocess_cfg,
        augment_cfg=augment_cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = create_dr_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    class_weights = make_class_weights(train_csv, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_cfg = TrainConfig(
        epochs=epochs,
        device=device,
        mixed_precision=device.startswith("cuda"),
        checkpoints_dir=cfg["output"]["checkpoints_dir"],
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=train_cfg,
    )

    # Save training history (optional but useful)
    out_dir = Path(cfg["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.yaml").write_text(yaml.safe_dump(history), encoding="utf-8")
    print("Saved history to:", out_dir / "history.yaml")


if __name__ == "__main__":
    main()