from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as tvm
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score

from src.data.data_colab import load_data_colab
from src.data.split_manager import load_splits, split_signature
from src.data.label_mapping import (
    CANONICAL_CLASSES,
    CANONICAL_CLASSES_FP,
    encode_labels,
    reorder_probs_to_canonical,
)
from src.export.model_exporter import export_predictions, load_predictions
from src.data.image_dataset import RakutenImageDataset
import wandb

@dataclass
class ResNet50Config:
    # Data / IO
    raw_dir: str                      # e.g. "/content/data/raw" (CSV location)
    img_dir: str                      # e.g. "/content/images/images/image_train" (can be unused if df has image_path)
    out_dir: str                      # e.g. "<STORE>/artifacts/exports"
    ckpt_dir: str                     # e.g. "<STORE>/checkpoints/image_resnet50"

    # Training
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 1
    lr: float = 1e-4
    use_amp: bool = True
    label_smoothing: float = 0.1
    dropout_rate: float = 0.3

    # Scheduler
    plateau_factor: float = 0.1
    plateau_patience: int = 3

    # Runtime
    device: Optional[str] = None      # "cuda" or "cpu"
    model_name: str = "resnet50_rerun_canonical"

    # Export split
    export_split: str = "val"         # "val" (recommended) or "test"


class IndexedDataset(Dataset):
    """
    Wrap a base dataset to return (image, label, real_idx) to preserve and verify alignment.
    This assumes `base_dataset[i]` is valid for i in [0, len(full_df)-1].
    """
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices).astype(int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real_idx = int(self.indices[i])
        img, label = self.base[real_idx]
        return img, label, real_idx


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def _build_resnet50(num_classes: int, dropout_rate: float) -> nn.Module:
    """
    Minimal-dependency ResNet50 using torchvision, with a dropout+linear head.
    """
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=float(dropout_rate)),
        nn.Linear(in_features, int(num_classes)),
    )
    return model


def _make_loaders(
    df_full: pd.DataFrame,
    y_encoded: np.ndarray,
    splits: Dict[str, np.ndarray],
    img_dir: Path,
    cfg: ResNet50Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test loaders with canonical split indices and canonical label ids.
    """
    df_full = df_full.copy()
    df_full["encoded_label"] = y_encoded.astype(int)

    train_df = df_full.iloc[splits["train_idx"]].reset_index(drop=True)
    val_df = df_full.iloc[splits["val_idx"]].reset_index(drop=True)
    test_df = df_full.iloc[splits["test_idx"]].reset_index(drop=True)

    train_tf, val_tf = _build_transforms(cfg.img_size)

    train_dataset = RakutenImageDataset(
        dataframe=train_df,
        image_dir=str(img_dir),
        transform=train_tf,
        label_col="encoded_label",
    )
    val_dataset = RakutenImageDataset(
        dataframe=val_df,
        image_dir=str(img_dir),
        transform=val_tf,
        label_col="encoded_label",
    )
    test_dataset = RakutenImageDataset(
        dataframe=test_df,
        image_dir=str(img_dir),
        transform=val_tf,
        label_col="encoded_label",
    )

    # Note: pin_memory is useful when using CUDA
    pin_memory = bool((cfg.device or "").startswith("cuda") or torch.cuda.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            assert scaler is not None
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labs)

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) else 0.0
    return float(avg_loss), float(acc), float(f1)


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Val", ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labs)

        pbar.set_postfix(loss=float(loss.item()))

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) else 0.0
    return float(avg_loss), float(acc), float(f1)


@torch.no_grad()
def _predict_probs_with_real_idx(
    model: nn.Module,
    base_dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities for a given list of real indices, returning (probs, idx).
    Ensures export alignment by keeping the exact indices order.
    """
    model.eval()

    indexed = IndexedDataset(base_dataset, indices)
    loader = DataLoader(
        indexed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    probs_list = []
    idx_list = []

    for images, _, real_idx in tqdm(loader, desc="ExportInference", ncols=100, leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_list.append(probs)
        idx_list.append(real_idx.detach().cpu().numpy())

    probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, len(CANONICAL_CLASSES)), dtype=np.float32)
    idx = np.concatenate(idx_list, axis=0) if idx_list else np.zeros((0,), dtype=int)
    return probs, idx


def run_resnet50_colab(cfg: ResNet50Config) -> Dict[str, Any]:
    """
    Colab-friendly canonical ResNet50 training + Phase 3 export + B4-verifiable contract.
    """

    wandb.init(
        project="rakuten_image",
        name=cfg.model_name,
        config=cfg,
        reinit=True
    )
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt_dir).expanduser().resolve()
    img_dir = Path(cfg.img_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load full data (NO split generation here)
    pack = load_data_colab(
    raw_dir=cfg.raw_dir,
    img_root=Path(cfg.img_dir),
    splitted=False,
    verbose=True,
    )
    X, y = pack["X"], pack["y"]


    # 2) Canonical splits (single source of truth)
    splits = load_splits(verbose=True)
    sig = split_signature(splits)

    # 3) Canonical label encoding (training IDs = canonical indices)
    y_encoded = encode_labels(y, CANONICAL_CLASSES).astype(int)

    # 4) DataLoaders
    train_loader, val_loader, _ = _make_loaders(
        df_full=X,
        y_encoded=y_encoded,
        splits=splits,
        img_dir=img_dir,
        cfg=cfg,
    )

    # 5) Model
    model = _build_resnet50(
        num_classes=len(CANONICAL_CLASSES),
        dropout_rate=cfg.dropout_rate,
    ).to(device)

    wandb.watch(model, log="all", log_freq=10)

    # 6) Optimization (Adam + ReduceLROnPlateau on val_f1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.plateau_factor,
        patience=cfg.plateau_patience,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # AMP (safe enabling)
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_path = ckpt_dir / "best_model.pth"
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    for epoch in range(int(cfg.num_epochs)):
        train_loss, train_acc, train_f1 = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler if use_amp else None,
        )

        val_loss, val_acc, val_f1 = _eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"])
        })

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1": best_val_f1,
                    "split_signature": sig,
                    "classes_fp": CANONICAL_CLASSES_FP,
                },
                best_path,
            )

    # Load best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 7) Export predictions (alignment-safe) on cfg.export_split
    _, val_transform = _build_transforms(cfg.img_size)

    full_df = X.copy()
    full_df["encoded_label"] = y_encoded

    full_dataset_for_export = RakutenImageDataset(
        dataframe=full_df.reset_index(drop=True),
        image_dir=str(img_dir),
        transform=val_transform,
        label_col="encoded_label",
    )

    export_idx = splits["val_idx"] if cfg.export_split == "val" else splits["test_idx"]

    # Export inference: set num_workers=0 for Colab stability
    probs, seen_idx = _predict_probs_with_real_idx(
        model=model,
        base_dataset=full_dataset_for_export,
        indices=export_idx,
        batch_size=cfg.batch_size,
        num_workers=0,
        device=device,
    )

    # Hard alignment check
    if not np.array_equal(seen_idx, export_idx):
        raise AssertionError("Index order mismatch during export inference")

    # Explicit no-op reorder for traceability
    probs_aligned = reorder_probs_to_canonical(probs, CANONICAL_CLASSES, CANONICAL_CLASSES)

    # y_true in original code space for the exported split
    y_true = y_encoded[seen_idx].astype(int)

    export_result = export_predictions(
        out_dir=out_dir,
        model_name=cfg.model_name,
        split_name=cfg.export_split,
        idx=seen_idx,
        split_signature=sig,
        probs=probs_aligned,
        classes=CANONICAL_CLASSES,
        y_true=y_true,
        extra_meta={
            "source": "src/train/image_resnet50.py",
            "model_architecture": "torchvision.resnet50",
            "img_dir": str(img_dir),
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "use_amp": use_amp,
            "label_smoothing": cfg.label_smoothing,
            "dropout_rate": cfg.dropout_rate,
            "classes_fp": CANONICAL_CLASSES_FP,
            "split_signature": sig,
            "export_split": cfg.export_split,
        },
    )

    # 8) Verify export contract (B4-compatible strict checks)
    loaded = load_predictions(
        npz_path=export_result["npz_path"],
        verify_split_signature=sig,
        verify_classes_fp=CANONICAL_CLASSES_FP,
        require_y_true=True,
    )

    wandb.finish()

    return {
        "export_result": export_result,
        "verify_metadata": loaded["metadata"],
        "probs_shape": loaded["probs"].shape,
        "best_val_f1": float(best_val_f1),
        "history": history,
    }
