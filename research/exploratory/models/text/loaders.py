from pathlib import Path
import torch
from transformers import AutoTokenizer

from .model import TextClassifier


def load_text_transformer(
    model_dir: str | Path,
    model_name: str,
    num_labels: int,
    mlp_dim: int = 512,
    pooling: str = "mean",
    device: str = "cpu",
):
    """
    Generic loader for CamemBERT / XLM-R text classifiers.

    Expected directory structure:
    model_dir/
        ├── model.pt          (state_dict or checkpoint with 'state_dict')
        └── tokenizer/

    Returns:
        model (nn.Module)
        tokenizer (AutoTokenizer)
    """
    model_dir = Path(model_dir).resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer",
        use_fast=True,
        local_files_only=True,
    )

    # --- Rebuild model architecture ---
    model = TextClassifier(
        model_name=model_name,
        num_labels=num_labels,
        mlp_dim=mlp_dim,
        pooling=pooling,
    )

    model.backbone.resize_token_embeddings(len(tokenizer) + 1)

    # --- Load state_dict ---
    checkpoint = torch.load(
        model_dir / "model.pt",
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer
