import torch
import numpy as np
from PIL import Image
from utils.calibration import calibrated_probas
from PIL import Image
import numpy as np
import torch


class ImageModelPipeline:
    """
    Generic inference pipeline for image classification models
    (ConvNeXt, Swin, ViT, etc.).

    This pipeline performs preprocessing, batched forward passes,
    and probability calibration. It operates fully in memory and
    does not require images to be stored on disk.
    """

    def __init__(
        self,
        model,
        transform,
        temperature: float,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        """
        Initialize the image model inference pipeline.

        Args:
            model (torch.nn.Module):
                Trained image classification model. The model must accept
                a batch of images and return logits of shape
                (batch_size, n_classes).

            transform (callable):
                Preprocessing transform applied to each image
                (e.g. torchvision.transforms.Compose).

            temperature (float):
                Temperature parameter used for probability calibration.

            device (str, optional):
                Device on which inference is performed ("cpu" or "cuda").
                Defaults to "cpu".

            batch_size (int, optional):
                Number of images processed per forward pass.
                Defaults to 16.
        """
        self.model = model
        self.transform = transform
        self.T = temperature
        self.device = device
        self.batch_size = batch_size

        self.model.to(device)
        self.model.eval()

    def _predict_logits(self, images) -> np.ndarray:
        """
        Compute raw logits for a batch of images.

        Args:
            images (list[PIL.Image.Image]):
                List of input images loaded in memory.

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes) containing
                the unnormalized logits.
        """
        if not images:
            raise ValueError("`images` must contain at least one image.")

        # Ensure PIL images
        processed_images = [
            img.convert("RGB") if isinstance(img, Image.Image) else img
            for img in images
        ]

        logits_all = []

        with torch.no_grad():
            for i in range(0, len(processed_images), self.batch_size):
                batch_imgs = processed_images[i:i + self.batch_size]

                batch_tensor = torch.stack([
                    self.transform(img) for img in batch_imgs
                ]).to(self.device)

                outputs = self.model(batch_tensor)
                logits_all.append(outputs.cpu().numpy())

        logits = np.vstack(logits_all)

        assert logits.shape[0] == len(images), (
            f"Mismatch between number of inputs ({len(images)}) "
            f"and outputs ({logits.shape[0]})."
        )

        return logits

    def predict_proba(self, images) -> np.ndarray:
        """
        Predict calibrated class probabilities for input images.

        Args:
            images (list[PIL.Image.Image]):
                List of input images loaded in memory.

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes) containing
                calibrated class probabilities.
        """
        logits = self._predict_logits(images)
        return calibrated_probas(logits, self.T)
