from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import confusion_matrix, classification_report

from src.data.augmentation import AugmentConfig, get_val_transforms
from src.data.preprocessing import PreprocessConfig, preprocess_image
from src.models.model import create_dr_model
from src.training.metrics import compute_classification_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--weights", type=str, default="models/best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    image_size = int(cfg["data"]["image_size"])
    images_dir = Path(cfg["data"]["images_dir"])
    num_classes = int(cfg["data"]["num_classes"])
    model_name = cfg["train"]["model_name"]

    split_csv = Path(f"data/splits/{args.split}.csv")
    df = pd.read_csv(split_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Model
    model = create_dr_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    preprocess_cfg = PreprocessConfig(image_size=image_size, crop_black=True, enhance_contrast=False)
    val_tfms = get_val_transforms(AugmentConfig(image_size=image_size))

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        img_path = images_dir / f"{row['id_code']}.png"
        label = int(row["diagnosis"])

        img = preprocess_image(img_path, preprocess_cfg)
        x = val_tfms(image=img)["image"].unsqueeze(0).to(device)  # [1,3,H,W]

        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

        y_true.append(label)
        y_pred.append(pred)

    metrics = compute_classification_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({args.split})")
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_matrix_{args.split}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()