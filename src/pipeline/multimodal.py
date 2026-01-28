import numpy as np
import pandas as pd
from data.label_mapping import CANONICAL_CLASSES, decode_labels


class MetaPipeline:
    """
    Multimodal inference pipeline based on a stacking meta-classifier.

    This pipeline combines text-based and image-based predictions by
    concatenating their probability outputs and feeding them to a
    trained meta-model (stacking).

    The pipeline operates fully in memory and does not require images
    to be stored on disk.
    """

    def __init__(
        self,
        meta_model,
        text_pipeline,
        image_pipeline,
        text_cols=("designation", "description"),
        classes=CANONICAL_CLASSES,
    ):
        """
        Initialize the multimodal meta-pipeline.

        Args:
            meta_model:
                Trained meta-classifier implementing `predict_proba(X)`.
                It must accept concatenated text and image probabilities.

            text_pipeline:
                Text inference pipeline exposing `predict_proba(texts)`.

            image_pipeline:
                Image inference pipeline exposing `predict_proba(images)`.

            text_cols (tuple[str], optional):
                Names of the text columns used to build textual inputs.
                Defaults to ("designation", "description").

            classes (list, optional):
                Canonical class labels corresponding to model outputs.
        """
        self.text_pipeline = text_pipeline
        self.image_pipeline = image_pipeline
        self.meta_model = meta_model
        self.text_cols = list(text_cols)
        self.classes = classes

    def predict_proba(self, texts, images):
        """
        Predict class probabilities using the multimodal stacking pipeline.

        Args:
            texts (array-like or pandas.DataFrame):
                Text inputs. If a DataFrame is provided, `text_cols`
                are extracted automatically.

            images (list[PIL.Image.Image]):
                List of images loaded in memory.

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes) with final
                class probabilities predicted by the meta-model.
        """
        texts_prepared = self._prepare_texts(texts)

        P_text = self.text_pipeline.predict_proba(texts_prepared)
        P_image = self.image_pipeline.predict_proba(images)

        X_meta = np.concatenate([P_text, P_image], axis=1)
        return self.meta_model.predict_proba(X_meta)

    def predict(self, texts, images):
        """
        Predict canonical class indices.

        Args:
            texts:
                Text inputs.

            images:
                Image inputs.

        Returns:
            np.ndarray:
                Predicted class indices (0..n_classes-1).
        """
        return np.argmax(self.predict_proba(texts, images), axis=1)

    def predict_labels(self, texts, images):
        """
        Predict original product type codes.

        Args:
            texts:
                Text inputs.

            images:
                Image inputs.

        Returns:
            list:
                Decoded product type labels.
        """
        y_pred = self.predict(texts, images)
        return decode_labels(y_pred, self.classes)

    def predict_with_contributions(self, texts, images):
        """
        Predict labels along with modality-specific contributions.

        For each sample, this method returns:
        - predicted label
        - final meta-model probability
        - text model probability for the predicted class
        - image model probability for the predicted class

        Args:
            texts:
                Text inputs.

            images:
                Image inputs.

        Returns:
            pandas.DataFrame:
                DataFrame with columns:
                - label_pred
                - P_final
                - P_text
                - P_image
        """
        texts_prepared = self._prepare_texts(texts)

        P_text = self.text_pipeline.predict_proba(texts_prepared)
        P_image = self.image_pipeline.predict_proba(images)

        X_meta = np.concatenate([P_text, P_image], axis=1)
        P_final = self.meta_model.predict_proba(X_meta)

        y_pred = np.argmax(P_final, axis=1)
        idx = np.arange(len(y_pred))

        labels_pred = decode_labels(y_pred, self.classes)

        return pd.DataFrame({
            "label_pred": labels_pred,
            "P_final": P_final[idx, y_pred],
            "P_text": P_text[idx, y_pred],
            "P_image": P_image[idx, y_pred],
        })

    def _prepare_texts(self, texts):
        """
        Prepare textual inputs for the text pipeline.

        Args:
            texts:
                Either a DataFrame containing text columns or
                a list/array of already prepared strings.

        Returns:
            list[str]:
                Prepared text inputs.
        """
        if hasattr(texts, "__getitem__") and all(col in texts for col in self.text_cols):
            return texts[self.text_cols]
        return texts
