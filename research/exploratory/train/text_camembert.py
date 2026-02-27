"""
CamemBERT Canonical Training + Phase 3 Export

Dependencies:
- transformers (pip install transformers)
- torch, numpy, pandas, sklearn, tqdm, datasets (standard ML stack)

Local usage:
    python -m src.train.text_camembert --raw-dir data/raw --out-dir artifacts/exports ...

Colab usage:
    python -m src.train.text_camembert --raw-dir /content/drive/.../data_raw ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

# FIXED: Colab loader always available, local loader optional
from src.data.data_colab import load_data_colab
import wandb

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
import wandb


@dataclass
class CamemBERTConfig:
    # Data / IO
    raw_dir: str                      # Path to raw CSV directory
    out_dir: str                      # Export output directory
    ckpt_dir: str                     # Checkpoint directory

    # Text columns
    text_col: str = "designation"     # Primary text column
    text_col2: str = "description"    # Secondary text column (concatenated)

    # Training
    max_length: int = 384
    batch_size: int = 64  # Optimized for Colab GPU
    num_epochs: int = 6
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    # Regularization
    label_smoothing: float = 0.0
    dropout: float = 0.15

    # Early stopping
    patience: int = 2

    # Model
    model_name: str = "almanach/camembert-base"  # or "almanach/camembert-large"

    # Data loader
    force_colab_loader: bool = False  # Force Colab loader (ignores local loader)

    # Runtime
    device: Optional[str] = None      # "cuda" or "cpu"
    export_name: str = "camembert_canonical"  # Export name

    # Export split
    export_split: str = "val"         # "val" (recommended) or "test"


def build_text_column(df: pd.DataFrame, col1: str = "designation", col2: str = "description") -> pd.Series:
    """
    Build combined text column from designation + description.
    Handles missing values safely.
    """
    def safe_str(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    parts = []
    if col1 in df.columns:
        parts.append(df[col1].map(safe_str))
    if col2 in df.columns:
        parts.append(df[col2].map(safe_str))

    if not parts:
        raise ValueError(f"Neither {col1} nor {col2} found in dataframe columns: {df.columns.tolist()}")

    # Join with space
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p

    return combined.str.strip()


def make_hf_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Dataset:
    """Convert pandas DataFrame to HuggingFace Dataset."""
    return Dataset.from_dict({
        "text": df[text_col].tolist(),
        "label": df[label_col].tolist(),
    })


def tokenize_fn(examples, tokenizer, max_length: int):
    """Tokenize batch of examples."""
    try:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False  # Not needed for RoBERTa-based models
        )
    except TypeError:
        # Fallback for older tokenizer versions
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length
        )


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def run_camembert_canonical(cfg: CamemBERTConfig) -> Dict[str, Any]:
    """
    Canonical CamemBERT training + Phase 3 export + B4-verifiable contract.
    Uses unified splits (data/splits/*.txt) to ensure alignment with image models.
    """
    wandb.init(
        project="rakuten_text",
        name=f"camembert",
        config=cfg.__dict__,
        reinit=True,
    )

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 1) Load full data (NO split generation here)
    # ======================================================================
    if cfg.force_colab_loader:
        print("[INFO] Using Colab data loader (forced via force_colab_loader=True)")
        pack = load_data_colab(
            raw_dir=cfg.raw_dir,
            img_root=None,  # Not needed for text
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
            img_root=None,
            splitted=False,
            verbose=True,
        )
        X, y = pack["X"], pack["y"]

    # ======================================================================
    # 2) ⚠️ CRITICAL: Load canonical splits (single source of truth)
    # ======================================================================
    splits = load_splits(verbose=True)
    sig = split_signature(splits)
    print(f"✓ Split signature: {sig}")
    print(f"✓ Split sizes: train={len(splits['train_idx'])}, val={len(splits['val_idx'])}, test={len(splits['test_idx'])}")

    # ======================================================================
    # 3) Build text column
    # ======================================================================
    X["text"] = build_text_column(X, cfg.text_col, cfg.text_col2)

    # Filter empty text (optional but recommended)
    valid_mask = X["text"].str.len() > 0
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"[WARNING] Filtering {n_invalid} rows with empty text")
        # This is where the issue occurs in NB08!
        # We must ensure indices still align with splits
        # Solution: Keep all rows, just replace empty with placeholder
        X.loc[~valid_mask, "text"] = "[EMPTY]"

    # ======================================================================
    # 4) Canonical label encoding (training IDs = canonical indices)
    # ======================================================================
    y_encoded = encode_labels(y, CANONICAL_CLASSES).astype(int)
    X["label"] = y_encoded

    # ======================================================================
    # 5) ⚠️ CRITICAL: Split data using canonical indices
    # ======================================================================
    train_df = X.iloc[splits["train_idx"]].copy().reset_index(drop=True)
    val_df = X.iloc[splits["val_idx"]].copy().reset_index(drop=True)
    test_df = X.iloc[splits["test_idx"]].copy().reset_index(drop=True)

    print(f"✓ Train size: {len(train_df)}")
    print(f"✓ Val size: {len(val_df)}")
    print(f"✓ Test size: {len(test_df)}")

    # ======================================================================
    # 6) Create HuggingFace Datasets
    # ======================================================================
    train_ds = make_hf_dataset(train_df, "text", "label")
    val_ds = make_hf_dataset(val_df, "text", "label")
    test_ds = make_hf_dataset(test_df, "text", "label")

    # ======================================================================
    # 7) Load tokenizer and model
    # ======================================================================
    print(f"[INFO] Loading tokenizer: {cfg.model_name}")
    use_fast = "flaubert" not in cfg.model_name.lower()  # FlauBERT doesn't have fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=use_fast)

    print(f"[INFO] Loading model: {cfg.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(CANONICAL_CLASSES),
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
    )

    # ======================================================================
    # 8) Tokenize datasets
    # ======================================================================
    print("[INFO] Tokenizing datasets...")
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=["text"]
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=["text"]
    )
    test_ds = test_ds.map(
        lambda x: tokenize_fn(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=["text"]
    )

    # Set format for PyTorch
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")
    test_ds.set_format(type="torch")

    # ======================================================================
    # 9) Training arguments
    # ======================================================================
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        logging_dir=str(ckpt_dir / "logs"),
        logging_steps=50,
        report_to=["wandb"],

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=10,  # Optimized for Colab
        dataloader_pin_memory=True,

        seed=42,
    )

    # ======================================================================
    # 10) Trainer
    # ======================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    # ======================================================================
    # 11) Train
    # ======================================================================
    print("[INFO] Starting training...")
    trainer.train()

    # ======================================================================
    # 12) Evaluate on validation set
    # ======================================================================
    print("[INFO] Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(f"✓ Validation metrics: {val_metrics}")

    # ======================================================================
    # 13) Export predictions (alignment-safe) on cfg.export_split
    # ======================================================================
    export_idx = splits["val_idx"] if cfg.export_split == "val" else splits["test_idx"]
    export_ds = val_ds if cfg.export_split == "val" else test_ds

    print(f"[INFO] Generating predictions for {cfg.export_split} set...")
    predictions = trainer.predict(export_ds)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    # Verify indices alignment
    if len(probs) != len(export_idx):
        raise AssertionError(f"Probs length ({len(probs)}) != export_idx length ({len(export_idx)})")

    # Explicit no-op reorder for traceability
    probs_aligned = reorder_probs_to_canonical(probs, CANONICAL_CLASSES, CANONICAL_CLASSES)

    # y_true must be canonical indices (0..26)
    y_true = y_encoded[export_idx].astype(int)

    # ======================================================================
    # 14) Export with signature verification
    # ======================================================================
    export_result = export_predictions(
        out_dir=out_dir,
        model_name=cfg.export_name,
        split_name=cfg.export_split,
        idx=export_idx,
        split_signature=sig,
        probs=probs_aligned,
        classes=CANONICAL_CLASSES,
        y_true=y_true,
        extra_meta={
            "source": "src/train/text_camembert.py",
            "model_architecture": cfg.model_name,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "dropout": cfg.dropout,
            "classes_fp": CANONICAL_CLASSES_FP,
            "split_signature": sig,
            "export_split": cfg.export_split,
        },
    )

    # ======================================================================
    # 15) Verify export contract (B4-compatible strict checks)
    # ======================================================================
    loaded = load_predictions(
        npz_path=export_result["npz_path"],
        verify_split_signature=sig,
        verify_classes_fp=CANONICAL_CLASSES_FP,
        require_y_true=True,
    )

    print(f"✓ Export verified: {loaded['metadata']['model_name']}")
    print(f"✓ Split signature matches: {loaded['metadata']['split_signature'] == sig}")
    print(f"✓ Classes fingerprint matches: {loaded['metadata']['classes_fp'] == CANONICAL_CLASSES_FP}")

    wandb.finish()

    return {
        "export_result": export_result,
        "verify_metadata": loaded["metadata"],
        "probs_shape": loaded["probs"].shape,
        "val_metrics": val_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CamemBERT Canonical Training + Export")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw CSV directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/exports", help="Export output directory")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/text_camembert", help="Checkpoint directory")

    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Max sequence length")

    # Model selection
    parser.add_argument("--model", type=str, default="almanach/camembert-base",
                        help="HuggingFace model name (default: almanach/camembert-base)")

    # Data loader selection
    parser.add_argument("--force-colab-loader", dest="force_colab_loader", action="store_true",
                        help="Force load_data_colab(raw_dir=...) (recommended in Colab)")
    parser.set_defaults(force_colab_loader=False)

    parser.add_argument("--export-name", type=str, default="camembert_canonical", help="Model name for export")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Export split")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CamemBERTConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_name=args.model,
        force_colab_loader=args.force_colab_loader,
        device=args.device,
        export_name=args.export_name,
        export_split=args.split,
    )

    print("="*80)
    print("CamemBERT Canonical Training Configuration")
    print("="*80)
    print(f"Raw dir: {cfg.raw_dir}")
    print(f"Export dir: {cfg.out_dir}")
    print(f"Checkpoint dir: {cfg.ckpt_dir}")
    print(f"Model: {cfg.model_name}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Max length: {cfg.max_length}")
    print(f"Export name: {cfg.export_name}")
    print(f"Export split: {cfg.export_split}")
    print("="*80)

    result = run_camembert_canonical(cfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Val metrics: {result['val_metrics']}")
    print(f"Probs shape: {result['probs_shape']}")
    print("\nExport Result:")
    for k, v in result["export_result"].items():
        print(f"  {k}: {v}")
    print("="*80)
