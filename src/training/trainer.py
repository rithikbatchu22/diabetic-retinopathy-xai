from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_classification_metrics


@dataclass
class TrainConfig:
    epochs: int = 5
    device: str = "cuda"
    mixed_precision: bool = True  # good for Tesla T4
    checkpoints_dir: str = "models"
    best_ckpt_name: str = "best_model.pth"
    last_ckpt_name: str = "last_checkpoint.pth"


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0

    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    val_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)
    return val_loss, metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: TrainConfig,
) -> dict:
    device = cfg.device
    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if (cfg.mixed_precision and device.startswith("cuda")) else None

    best_qwk = -1.0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    ckpt_dir = Path(cfg.checkpoints_dir)
    best_path = ckpt_dir / cfg.best_ckpt_name
    last_path = ckpt_dir / cfg.last_ckpt_name

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        # Always save last checkpoint (resume-safe for Colab)
        _save_checkpoint(last_path, {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_qwk": best_qwk,
        })

        # Save best model by QWK
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            _save_checkpoint(best_path, {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_qwk": best_qwk,
                "val_metrics": val_metrics,
            })

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | f1={val_metrics['macro_f1']:.4f} | qwk={val_metrics['qwk']:.4f}"
        )

    return history