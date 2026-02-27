import torch
from pathlib import Path
from train.image_swin import RakutenSwin
from train.image_convnext import RakutenConvNeXt


def load_convnext_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    num_classes: int = 27,
    model_name: str = "convnext_base",
    drop_path_rate: float = 0.3,
    dropout_rate: float = 0.5,
    head_dropout2: float = 0.3,
):
    checkpoint_path = Path(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint invalide : 'model_state_dict' manquant")

    model = RakutenConvNeXt(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        drop_path_rate=drop_path_rate,
        dropout_rate=dropout_rate,
        head_dropout2=head_dropout2,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_swin_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    num_classes: int = 27,
    model_name: str = "swin_base_patch4_window7_224",
    drop_path_rate: float = 0.3,
    dropout_rate: float = 0.5,
    head_dropout2: float = 0.3,
):
    checkpoint_path = Path(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint invalide : 'model_state_dict' manquant")

    model = RakutenSwin(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        drop_path_rate=drop_path_rate,
        dropout_rate=dropout_rate,
        head_dropout2=head_dropout2,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model
