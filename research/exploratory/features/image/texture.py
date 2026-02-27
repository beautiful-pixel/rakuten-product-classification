import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin


class HOGTransformer(BaseEstimator, TransformerMixin):
    """
    Extracteur de descripteurs HOG (Histogram of Oriented Gradients).

    Ce transformateur calcule des histogrammes d'orientations de gradients
    afin de représenter la texture et la structure locale des images.
    Les descripteurs HOG sont couramment utilisés pour la reconnaissance
    d’objets et l’analyse de formes.

    Args:
        orientations (int): Nombre de bins d’orientation des gradients.
        pixels_per_cell (tuple[int, int]): Taille (en pixels) d’une cellule.
        cells_per_block (tuple[int, int]): Nombre de cellules par bloc.

    Returns:
        np.ndarray:
            Tableau de forme (n_samples, n_features) contenant les descripteurs
            HOG calculés pour chaque image.
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (3, 3),
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        """
        Méthode factice requise pour la compatibilité sklearn.

        Args:
            X (np.ndarray): Images d’entrée.
            y (np.ndarray, optional): Labels.

        Returns:
            HOGTransformer: Instance du transformateur.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Calcule les descripteurs HOG pour chaque image.

        Args:
            X (np.ndarray): Tableau d’images de forme
                (n_samples, height, width) ou
                (n_samples, height, width, channels).

        Returns:
            np.ndarray:
                Descripteurs HOG pour chaque image.
        """
        features = np.array([
            hog(
                X[i],
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block
            )
            for i in range(len(X))
        ])
        return features
