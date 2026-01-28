from pathlib import Path
import torch
from safetensors.torch import load_file as load_safetensors

def load_state_dict(model_dir: Path, device: str = "cpu"):
    """
    Load a model state_dict from either a safetensors file or a PyTorch checkpoint.

    Priority:
    1. model.safetensors
    2. model.pt

    Args:
        model_dir (Path): Directory containing model weights.
        device (str): Target device.

    Returns:
        dict: state_dict ready to be loaded into the model.
    """

    safetensors_path = model_dir / "model.safetensors"
    pt_path = model_dir / "model.pt"

    if safetensors_path.exists():
        state_dict = load_safetensors(safetensors_path)
        return state_dict

    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        return checkpoint.get("state_dict", checkpoint)

    raise FileNotFoundError(
        f"No model weights found in {model_dir}. "
        "Expected 'model.safetensors' or 'model.pt'."
    )