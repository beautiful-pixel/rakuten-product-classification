import numpy as np

class BlendingPipeline:
    def __init__(self, models: dict, weights: dict, device: str = "cpu"):
        """
        Generic blending (late-fusion) pipeline using weighted probability averaging.

        This pipeline is agnostic to the input modality (text, image, etc.).
        Each model must expose a `predict_proba(inputs)` method and return
        probabilities with shape (n_samples, n_classes).

        Args:
            models (dict):
                Dictionary mapping model names to inference pipelines.

            weights (dict):
                Dictionary mapping model names to blending weights.

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
                f"Missing blending weights for models: {sorted(missing_weights)}"
            )

        if extra_weights:
            raise KeyError(
                f"Blending weights provided for unknown models: {sorted(extra_weights)}"
            )

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Blending weights must sum to a positive value.")

        # Normalize weights once
        self.weights = {k: v / total_weight for k, v in weights.items()}
        self.models = models
        self.device = device

    def predict_proba(self, inputs):
        """
        Predict class probabilities using weighted blending.

        Args:
            inputs:
                Model inputs (e.g. list[str] for text, list[PIL.Image] for images).

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes) with blended probabilities.
        """

        probs = []

        for name, model in self.models.items():
            p = model.predict_proba(inputs)
            probs.append(self.weights[name] * p)

        probs = np.stack(probs, axis=0).sum(axis=0)
        return probs / probs.sum(axis=1, keepdims=True)