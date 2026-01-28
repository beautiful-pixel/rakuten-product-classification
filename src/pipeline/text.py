# src/pipeline/text.py

import numpy as np
import torch

import json
import joblib

from utils.calibration import calibrated_probas
from models.text.loaders import load_text_transformer
from features.text import NumericTokensTransformer, MergeTextTransformer

    
class TransformerTextPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        tokenizer_params,
        temperature: float,
        device: str = "cpu",
        preprocess = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.T = temperature
        self.device = device
        self.preprocess = preprocess

    def _predict_logits(self, texts, batch_size: int = 32):
        if self.preprocess:
            texts = self.preprocess.transform(texts)

        self.model.eval()
        logits_all = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                enc = self.tokenizer(
                    list(batch_texts),
                    return_tensors="pt",
                    **self.tokenizer_params,
                ).to(self.device)

                outputs = self.model(**enc)
                logits = outputs.logits.cpu().numpy()
                logits_all.append(logits)

        return np.vstack(logits_all)

    def predict_proba(self, texts):
        logits = self._predict_logits(texts)
        return calibrated_probas(logits, self.T)
    

class TextFusionPipeline:
    def __init__(self, models: dict, weights: dict, device: str = "cpu"):
        """
        Late-fusion pipeline for text models using weighted probability averaging.

        Args:
            models (dict):
                Dictionary mapping model names to inference pipelines.
                Each model must expose a `predict_proba(texts)` method.

            weights (dict):
                Dictionary mapping model names to fusion weights.

            device (str, optional):
                Inference device (kept for interface consistency).
        """

        if not models:
            raise ValueError("`models` cannot be empty.")

        if not weights:
            raise ValueError("`weights` cannot be empty.")

        model_names = set(models.keys())
        weight_names = set(weights.keys())

        missing_weights = model_names - weight_names
        extra_weights = weight_names - model_names

        if missing_weights:
            raise KeyError(
                f"Missing fusion weights for models: {sorted(missing_weights)}"
            )

        if extra_weights:
            raise KeyError(
                f"Fusion weights provided for unknown models: {sorted(extra_weights)}"
            )

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Fusion weights must sum to a positive value.")

        # Normalize weights once at initialization
        self.weights = {k: v / total_weight for k, v in weights.items()}
        self.models = models
        self.device = device

    def predict_proba(self, texts):
        """
        Predict class probabilities using weighted late fusion.

        Args:
            texts (list[str]):
                Input texts to classify.

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes) with normalized probabilities.
        """

        probs = []

        for name, model in self.models.items():
            p = model.predict_proba(texts)
            probs.append(self.weights[name] * p)

        probs = np.stack(probs, axis=0).sum(axis=0)
        return probs / probs.sum(axis=1, keepdims=True)