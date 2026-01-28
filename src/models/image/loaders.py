import yaml
from pathlib import Path
from models.image.model import TimmImageClassifier
from utils.io import load_state_dict
from torchvision import transforms



def load_image_model(model_dir: str | Path, device: str = "cpu"):
    """
    Load an image classification model and its inference-time transforms
    from a local model directory.

    This function reconstructs the full image inference stack using
    configuration files versioned alongside the model weights. It ensures
    that the architecture, pretrained backbone parameters, and image
    preprocessing pipeline are strictly identical to those used during
    training.

    The function supports multiple image backbones (e.g. ConvNeXt, Swin)
    through a transform registry defined by the model configuration.

    Args:
        model_dir (str | Path):
            Path to the directory containing the image model artifacts.
            The directory is expected to follow the structure:

            ```
            model_dir/
                ├── model.pt or model.safetensors   # model weights
                └── config.yaml                     # model & transform config
            ```

        device (str, optional):
            Device on which the model should be loaded.
            Typical values are `"cpu"` or `"cuda"`.
            Defaults to `"cpu"`.

    Returns:
        dict:
            A dictionary containing the components required for image inference:

            - **model** (`torch.nn.Module`):
              Instantiated `TimmImageClassifier` with trained weights loaded,
              moved to the specified device, and set to evaluation mode.

            - **transform** (`torchvision.transforms.Compose`):
              Image preprocessing pipeline to apply before inference.
              This transform is built dynamically from the configuration
              and matches exactly the preprocessing used during training.

    Raises:
        ValueError:
            If the transform type specified in the configuration is unknown.

        FileNotFoundError:
            If no valid model weights (`model.pt` or `model.safetensors`)
            are found in the model directory.

    Notes:
        - Dropout and stochastic depth parameters are part of the model
          architecture and must be provided at load time. They are
          automatically disabled during inference via `model.eval()`.
        - The preprocessing pipeline is considered part of the model
          contract and must not be modified at inference time.
        - This function is intended to be called once at application
          startup (e.g. FastAPI or Streamlit initialization), not per request.
    """

    model_dir = Path(model_dir)

    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = TimmImageClassifier(
        model_name=cfg["backbone"]["model_name"],
        num_classes=cfg["head"]["num_classes"],
        pretrained=cfg["backbone"]["pretrained"],
        drop_path_rate=cfg["backbone"]["drop_path_rate"],
        hidden_dim=cfg["head"]["hidden_dim"],
        dropout=cfg["head"]["dropout"],
        head_dropout2=cfg["head"]["head_dropout2"],
        global_pool=cfg["backbone"].get("global_pool", "avg"),
    )

    state_dict = load_state_dict(model_dir, device=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    IMAGE_TRANSFORMS = {
        "convnext": lambda cfg: transforms.Compose([
            transforms.Resize(int(cfg["img_size"] * 1.14)),
            transforms.CenterCrop(cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["normalize"]["mean"],
                std=cfg["normalize"]["std"],
            ),
        ]),
        "swin": lambda cfg: transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["normalize"]["mean"],
                std=cfg["normalize"]["std"],
            ),
        ]),
    }

    t_cfg = cfg["transform"]
    t_type = t_cfg["type"]
    if t_type not in IMAGE_TRANSFORMS:
        raise ValueError(
            f"Unknown transform type '{t_type}'. "
            f"Available: {list(IMAGE_TRANSFORMS.keys())}"
        )
    transform = IMAGE_TRANSFORMS[t_type](t_cfg)

    return {
        "model" : model,
        "transform" : transform
    }
