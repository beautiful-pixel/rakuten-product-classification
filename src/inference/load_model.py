import yaml
import json
import joblib
from pathlib import Path
import torch

from pipeline.text import TransformerTextPipeline
from pipeline.image import ImageModelPipeline
from pipeline.blending import BlendingPipeline
from pipeline.multimodal import MetaPipeline
from models.text.loaders import load_text_transformer
from models.image.loaders import load_image_model
from utils.hf import hf_path, resolve_dir


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"

def load_model(
    repo_id: str,
    device: str = None,
):
    """
    Load the full multimodal inference pipeline (text + image + fusion)
    from Hugging Face Hub.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    registry_path = CONFIG_DIR / "registry.yaml"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    fusion_params_path = hf_path(
        repo_id=repo_id,
        filename=registry["fusion"]["params_path"]
    )

    with open(fusion_params_path) as f:
        fusion_params = json.load(f)

    fusion_text = fusion_params["text"]
    fusion_image = fusion_params["image"]

    text_models = {}

    for name, cfg in registry["text"].items():
        if name == "tfidf":
            continue

        model_dir = resolve_dir(
            repo_id=repo_id,
            path=cfg["path"]
        )

        text_bundle = load_text_transformer(
            model_dir=model_dir,
            device=device,
        )

        text_models[name] = TransformerTextPipeline(
            model=text_bundle['model'],
            tokenizer=text_bundle['tokenizer'],
            tokenizer_params=text_bundle['tokenizer_params'],
            temperature=fusion_text['temperatures'][name],
            preprocess=text_bundle['preprocess'],
            device=device,
        )

    tfidf_path = hf_path(
        repo_id=repo_id,
        filename=registry["text"]["tfidf"]["path"],
    )

    text_models["tfidf"] = joblib.load(tfidf_path)

    text_pipeline = BlendingPipeline(
        models=text_models,
        weights=fusion_text['weights']
    )

    image_models = {}

    for name, cfg in registry["image"].items():

        model_dir = resolve_dir(
            repo_id=repo_id,
            path=cfg["path"]
        )

        image_bundle = load_image_model(
            model_dir=model_dir,
            device=device,
        )

        image_models[name] = ImageModelPipeline(
            model=image_bundle['model'],
            transform=image_bundle['transform'],
            temperature=fusion_image['temperatures'][name],
            device=device,
        )

    image_pipeline = BlendingPipeline(
        models=image_models,
        weights=fusion_image['weights']
    )

    meta_model_path = hf_path(
        repo_id=repo_id,
        filename=registry["fusion"]["meta_model_path"],
    )

    meta_model = joblib.load(meta_model_path)

    model = MetaPipeline(
        meta_model=meta_model,
        text_pipeline=text_pipeline,
        image_pipeline=image_pipeline
    )


    return model