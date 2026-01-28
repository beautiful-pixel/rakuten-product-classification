from pathlib import Path
import yaml
from transformers import AutoTokenizer
from models.text.model import TransformerTextClassifier
from utils.io import load_state_dict
from features.text import NumericTokensTransformer, MergeTextTransformer




def load_text_transformer(
    model_dir: str | Path,
    device: str = "cpu",
):
    """
    Load a trained transformer-based text classification model and all
    its inference-time dependencies from a local model directory.

    This function reconstructs the full inference stack for a text model
    (e.g. CamemBERT or XLM-R) using artifacts versioned alongside the model
    weights. The model configuration, tokenizer parameters, and preprocessing
    strategy are read from the model's `config.yaml` file to guarantee full
    consistency between training and inference.

    The returned components are intended to be consumed by a higher-level
    inference pipeline (e.g. `TransformerTextPipeline`) and not used directly
    for prediction.

    Args:
        model_dir (str | Path):
            Path to the local directory containing the model artifacts.
            The directory is expected to follow the structure:

            ```
            model_dir/
                ├── model.pt              # PyTorch state_dict or checkpoint
                ├── config.yaml           # Model and tokenizer configuration
                └── tokenizer/            # Hugging Face tokenizer files
            ```

        device (str, optional):
            Device on which the model should be loaded.
            Typical values are `"cpu"` or `"cuda"`.
            Defaults to `"cpu"`.

    Returns:
        dict:
            A dictionary containing all components required for inference:

            - **model** (`torch.nn.Module`):
              Instantiated `TransformerTextClassifier` with trained weights
              loaded and set to evaluation mode.

            - **tokenizer** (`transformers.PreTrainedTokenizer`):
              Hugging Face tokenizer loaded from the local tokenizer directory.

            - **tokenizer_params** (`dict`):
              Dictionary of tokenizer runtime parameters extracted from
              `config.yaml` (e.g. `max_length`, `truncation`, `padding`).
              These parameters must be passed to the tokenizer at inference
              time to ensure consistency with training.

            - **preprocess** (`sklearn.base.TransformerMixin | None`):
              Optional scikit-learn compatible text preprocessing transformer
              (e.g. numeric token normalization or text merging), instantiated
              based on the preprocessing configuration stored in `config.yaml`.
              If no preprocessing is defined, this value is `None`.

    Notes:
        - Tokenizer parameters and preprocessing are considered part of the
          model contract when they affect the tokenizer vocabulary or input
          representation. They are therefore versioned together with the model
          artifacts.
        - The model weights are loaded with `strict=False` to allow backward
          compatibility with checkpoints that may include unused components
          (e.g. Hugging Face pooler layers).
        - This function does not perform inference itself and should not be
          called per request in production. It is intended to be executed once
          at application startup (e.g. FastAPI startup event).

    """

    model_dir = Path(model_dir).resolve()

    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer",
        use_fast=True,
        local_files_only=True,
    )

    TOKENIZER_KEYS = ["max_length", "truncation", "padding"]

    tokenizer_params = {
        k: v for k, v in cfg["tokenizer"].items()
        if k in TOKENIZER_KEYS
    }

    model = TransformerTextClassifier(
        model_name=cfg["backbone"]["model_name"],
        num_labels=cfg["head"]["num_labels"],
        mlp_dim=cfg["head"].get("mlp_dim", 512),
        pooling=cfg["head"].get("pooling", "mean"),
    )

    model.backbone.resize_token_embeddings(len(tokenizer) + 1)


    state_dict = load_state_dict(model_dir, device=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    TEXT_PREPROCESSORS = {
        "numeric_light": NumericTokensTransformer(strategy="light"),
        "merge_sep": MergeTextTransformer(sep="[SEP]"),
    }

    preprocess_type = cfg.get("preprocessing")
    preprocess = TEXT_PREPROCESSORS.get(preprocess_type)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "tokenizer_params": tokenizer_params,
        "preprocess": preprocess,
    }