"""
ConvNeXt Canonical Training + Phase 3 Export

Dependencies:
- timm (pip install timm) - Required for ConvNeXt models
- torch, torchvision, numpy, pandas, sklearn, tqdm (standard ML stack)

Local usage:
    python -m src.train.image_convnext --raw-dir data/raw --img-dir data/raw/images/image_train ...

Colab usage:
    python -m src.train.image_convnext --raw-dir /content/drive/.../data_raw --img-dir /content/images/... ...
"""
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
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score

# FIXED: Colab loader always available, local loader optional
from src.data.data_colab import load_data_colab

try:
    from src.data.data import load_data
    _USE_LOCAL_LOADER = True
except ImportError:
    _USE_LOCAL_LOADER = False

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
class ConvNeXtConfig:
    # Data / IO
    raw_dir: str                      # Path to raw CSV directory
    img_dir: str                      # Path to image directory
    out_dir: str                      # Export output directory
    ckpt_dir: str                     # Checkpoint directory

    # Training
    img_size: int = 384               # Higher resolution for ConvNeXt
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    use_amp: bool = True

    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.5          # Head dropout
    head_dropout2: float = 0.3         # Second head dropout
    drop_path_rate: float = 0.3        # Stochastic depth

    # Mixup/CutMix
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Scheduler
    cosine_eta_min: float = 1e-6

    # Model
    convnext_model_name: str = "convnext_base"
    convnext_pretrained: bool = True

    # Data loader
    force_colab_loader: bool = False  # Force Colab loader (ignores local loader)

    # Runtime
    device: Optional[str] = None      # "cuda" or "cpu"
    model_name: str = "convnext"      # Export name

    # Export split
    export_split: str = "val"         # "val" (recommended) or "test"


class IndexedDataset(Dataset):
    """
    Wrap a full dataset with specific indices to preserve and verify alignment.
    base_dataset must support indexing by full_df row numbers.
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


class RakutenConvNeXt(nn.Module):
    """ConvNeXt for Rakuten classification with custom head and stochastic depth."""

    def __init__(
        self,
        model_name: str = "convnext_base",
        num_classes: int = 27,
        pretrained: bool = True,
        drop_path_rate: float = 0.3,
        dropout_rate: float = 0.5,
        head_dropout2: float = 0.3,
    ):
        super(RakutenConvNeXt, self).__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ConvNeXt models.\n"
                "Install with: pip install timm\n"
                "Colab: !pip install timm\n"
                "Windows/Linux: pip install timm"
            )

        # ConvNeXt backbone without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        feature_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=head_dropout2),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes
        self.model_name = model_name

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation transforms for ConvNeXt with higher resolution."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Higher base size for center crop (438 -> 384)
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 438 for 384
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def _build_convnext(
    num_classes: int,
    cfg: ConvNeXtConfig,
) -> RakutenConvNeXt:
    """
    Build ConvNeXt with custom head for canonical classes.

    Args:
        num_classes: Number of output classes (27 for canonical)
        cfg: ConvNeXtConfig with model parameters

    Returns:
        RakutenConvNeXt model

    Raises:
        ImportError: If timm is not installed (pip install timm)
    """
    model = RakutenConvNeXt(
        model_name=cfg.convnext_model_name,
        num_classes=int(num_classes),
        pretrained=cfg.convnext_pretrained,
        drop_path_rate=cfg.drop_path_rate,
        dropout_rate=cfg.dropout_rate,
        head_dropout2=cfg.head_dropout2,
    )
    return model


def _make_loaders(
    df_full: pd.DataFrame,
    y_encoded: np.ndarray,
    splits: Dict[str, np.ndarray],
    img_dir: Path,
    cfg: ConvNeXtConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset]:
    """
    Build train/val/test loaders with canonical split indices and canonical label ids.
    Returns (train_loader, val_loader, test_loader, full_dataset) where full_dataset
    can be reused for export to maintain idx alignment.

    FIXED: Use full_dataset + IndexedDataset to preserve global idx semantics.
    """
    df_full = df_full.copy()
    df_full["encoded_label"] = y_encoded.astype(int)

    train_tf, val_tf = _build_transforms(cfg.img_size)

    # Build FULL dataset for training (will be wrapped with IndexedDataset)
    full_dataset_train = RakutenImageDataset(
        dataframe=df_full.reset_index(drop=True),
        image_dir=str(img_dir),
        transform=train_tf,
        label_col="encoded_label",
    )

    # Build FULL dataset for val/test (deterministic transform)
    full_dataset_val = RakutenImageDataset(
        dataframe=df_full.reset_index(drop=True),
        image_dir=str(img_dir),
        transform=val_tf,
        label_col="encoded_label",
    )

    pin_memory = bool((cfg.device or "").startswith("cuda") or torch.cuda.is_available())

    # FIXED: Use IndexedDataset to wrap full_dataset with split indices
    train_indexed = IndexedDataset(full_dataset_train, splits["train_idx"])
    val_indexed = IndexedDataset(full_dataset_val, splits["val_idx"])
    test_indexed = IndexedDataset(full_dataset_val, splits["test_idx"])

    train_loader = DataLoader(
        train_indexed,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_indexed,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_indexed,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, full_dataset_val


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
    mixup_fn: Optional[Any],
    model_ema: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Train one epoch with Mixup/CutMix and optional EMA support."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # FIXED: device_type for AMP autocast
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for batch in pbar:
        # FIXED: Handle IndexedDataset return (img, label, real_idx)
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Mixup/CutMix
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            assert scaler is not None
            # FIXED: device_type dynamic
            with torch.autocast(device_type=device_type, enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Update EMA if enabled
        if model_ema is not None:
            model_ema.update(model)

        total_loss += float(loss.item()) * images.size(0)

        # For metrics, we skip when using mixup (soft labels)
        if mixup_fn is None:
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs)

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0
    return float(avg_loss), float(acc), float(f1)


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """Evaluate one epoch without mixup."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    pbar = tqdm(loader, desc="Val", ncols=100, leave=False)
    for batch in pbar:
        # FIXED: Handle IndexedDataset return
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type=device_type, enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
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


def run_convnext_canonical(cfg: ConvNeXtConfig) -> Dict[str, Any]:
    """
    Canonical ConvNeXt training + Phase 3 export + B4-verifiable contract.
    Includes Mixup/CutMix, AdamW, CosineAnnealingLR, and optional EMA.
    """
    wandb.init(
        project="rakuten_image",
        name=f"convnext",
        config=cfg,
        reinit=True,
    )

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt_dir).expanduser().resolve()
    img_dir = Path(cfg.img_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load full data (NO split generation here)
    # FIXED: Force Colab loader when flag is set
    if cfg.force_colab_loader:
        print("[INFO] Using Colab data loader (forced via force_colab_loader=True)")
        pack = load_data_colab(
            raw_dir=cfg.raw_dir,
            img_root=Path(cfg.img_dir),
            splitted=False,
            verbose=True,
        )
        X, y = pack["X"], pack["y"]
    elif _USE_LOCAL_LOADER:
        print("[INFO] Using local data loader (src.data.data.load_data)")
        pack = load_data(splitted=False)
        X, y = pack["X"], pack["y"]
    else:
        print("[INFO] Using Colab data loader (src.data.data_colab.load_data_colab)")
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

    # 4) DataLoaders (FIXED: returns full_dataset for export reuse)
    train_loader, val_loader, _, full_dataset_val = _make_loaders(
        df_full=X,
        y_encoded=y_encoded,
        splits=splits,
        img_dir=img_dir,
        cfg=cfg,
    )

    # 5) Model
    model = _build_convnext(
        num_classes=len(CANONICAL_CLASSES),
        cfg=cfg,
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    # 6) EMA (Exponential Moving Average)
    model_ema = None
    if cfg.use_ema:
        try:
            from timm.utils import ModelEmaV2
            model_ema = ModelEmaV2(model, decay=cfg.ema_decay)
            print(f"[INFO] EMA initialized with decay={cfg.ema_decay}")
        except ImportError:
            print("[WARNING] timm.utils.ModelEmaV2 not available, EMA disabled")
            cfg.use_ema = False

    # 7) Mixup/CutMix
    try:
        from timm.data.mixup import Mixup
        from timm.loss import SoftTargetCrossEntropy

        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=cfg.mixup_switch_prob,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=len(CANONICAL_CLASSES),
        )
        criterion_train = SoftTargetCrossEntropy()
    except ImportError:
        raise ImportError(
            "timm.data.mixup and timm.loss are required for ConvNeXt training.\n"
            "Install with: pip install timm\n"
            "Colab: !pip install timm\n"
            "Windows/Linux: pip install timm"
        )

    criterion_val = nn.CrossEntropyLoss()

    # 8) Optimization (AdamW + CosineAnnealingLR)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=cfg.cosine_eta_min,
    )

    # AMP (safe enabling)
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_f1 = -1.0
    best_path = ckpt_dir / "best_model.pth"
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "ema_val_acc": [],
        "ema_val_f1": [],
        "lr": [],
    }

    for epoch in range(int(cfg.num_epochs)):
        train_loss, train_acc, train_f1 = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion_train,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            mixup_fn=mixup_fn,
            model_ema=model_ema,
        )

        val_loss, val_acc, val_f1 = _eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion_val,
            device=device,
            use_amp=use_amp,
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

        # Evaluate EMA model if enabled
        ema_val_acc, ema_val_f1 = 0.0, 0.0
        if model_ema is not None:
            _, ema_val_acc, ema_val_f1 = _eval_one_epoch(
                model=model_ema.module,
                loader=val_loader,
                criterion=criterion_val,
                device=device,
                use_amp=use_amp,
            )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["ema_val_acc"].append(ema_val_acc)
        history["ema_val_f1"].append(ema_val_f1)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if model_ema is not None:
            print(f"  EMA: val_acc={ema_val_acc:.4f} val_f1={ema_val_f1:.4f}")

        # Save best model (prefer EMA if better)
        current_f1 = max(val_f1, ema_val_f1)
        if current_f1 > best_val_f1:
            best_val_f1 = float(current_f1)
            use_ema_for_export = (ema_val_f1 > val_f1) and (model_ema is not None)
            torch.save(
                {
                    "model_state_dict": model_ema.module.state_dict() if use_ema_for_export else model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1": best_val_f1,
                    "split_signature": sig,
                    "classes_fp": CANONICAL_CLASSES_FP,
                    "is_ema": use_ema_for_export,
                },
                best_path,
            )

    # Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # 9) Export predictions (alignment-safe) on cfg.export_split
    export_idx = splits["val_idx"] if cfg.export_split == "val" else splits["test_idx"]

    # FIXED: Reuse full_dataset_val for export
    probs, seen_idx = _predict_probs_with_real_idx(
        model=model,
        base_dataset=full_dataset_val,
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

    # FIXED: y_true must be canonical indices (0..26), not original labels
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
            "source": "src/train/image_convnext.py",
            "model_architecture": f"timm.{cfg.convnext_model_name}",
            "convnext_pretrained": cfg.convnext_pretrained,
            "img_dir": str(img_dir),
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "use_amp": use_amp,
            "label_smoothing": cfg.label_smoothing,
            "drop_path_rate": cfg.drop_path_rate,
            "dropout_rate": cfg.dropout_rate,
            "mixup_alpha": cfg.mixup_alpha,
            "cutmix_alpha": cfg.cutmix_alpha,
            "use_ema": cfg.use_ema,
            "ema_decay": cfg.ema_decay,
            "classes_fp": CANONICAL_CLASSES_FP,
            "split_signature": sig,
            "export_split": cfg.export_split,
        },
    )

    # 10) Verify export contract (B4-compatible strict checks)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvNeXt Canonical Training + Export")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw CSV directory")
    parser.add_argument("--img-dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/exports", help="Export output directory")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/image_convnext", help="Checkpoint directory")

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")

    # Model selection and pretrained control
    parser.add_argument("--model", type=str, default="convnext_base",
                        help="timm model name (default: convnext_base)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Do not use pretrained weights")

    # Data loader selection
    parser.add_argument("--force-colab-loader", dest="force_colab_loader", action="store_true",
                        help="Force load_data_colab(raw_dir=...) (recommended in Colab)")
    parser.add_argument("--no-force-colab-loader", dest="force_colab_loader", action="store_false",
                        help="Do not force Colab loader (use local loader if available)")
    parser.set_defaults(force_colab_loader=False)

    parser.add_argument("--export-name", type=str, default="convnext", help="Model name for export")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Export split")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ConvNeXtConfig(
        raw_dir=args.raw_dir,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        img_size=384,  # Higher resolution for ConvNeXt
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=True,
        label_smoothing=0.1,
        dropout_rate=0.5,
        head_dropout2=0.3,
        drop_path_rate=0.3,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        use_ema=True,
        ema_decay=0.9999,
        convnext_model_name=args.model,
        convnext_pretrained=args.pretrained,
        force_colab_loader=args.force_colab_loader,
        device=args.device,
        model_name=args.export_name,
        export_split=args.split,
    )

    print("="*80)
    print("ConvNeXt Canonical Training Configuration")
    print("="*80)
    print(f"Raw dir: {cfg.raw_dir}")
    print(f"Image dir: {cfg.img_dir}")
    print(f"Export dir: {cfg.out_dir}")
    print(f"Checkpoint dir: {cfg.ckpt_dir}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Model name: {cfg.model_name}")
    print(f"Export split: {cfg.export_split}")
    print(f"EMA: {cfg.use_ema}")
    print("="*80)

    result = run_convnext_canonical(cfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best val F1: {result['best_val_f1']:.4f}")
    print(f"Probs shape: {result['probs_shape']}")
    print("\nExport Result:")
    for k, v in result["export_result"].items():
        print(f"  {k}: {v}")
    print("="*80)
