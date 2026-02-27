import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MinMaxDiffTransformer(BaseEstimator, TransformerMixin):
    """
    Extrait une feature basée sur les variations locales maximales d'un signal.

    Pour chaque intervalle défini dans `max_diff_ranges`, le transformateur
    calcule la différence maximale entre valeurs consécutives, puis conserve
    le minimum de ces maxima sur l'ensemble des intervalles.

    Cette feature permet de capturer une mesure robuste de variation
    (contraste, discontinuité) sur différentes plages du signal.
    """

    def __init__(self, max_diff_ranges=None):
        """
        Initialise le transformateur.

        Args:
            max_diff_ranges (list[list[int, int]], optional): Liste d'intervalles
                [début, fin[ sur lesquels calculer les différences.
                Par défaut, trois intervalles consécutifs de taille 256.
        """
        if max_diff_ranges is None:
            self.max_diff_ranges = [[i * 256, (i + 1) * 256] for i in range(3)]
        else:
            self.max_diff_ranges = max_diff_ranges

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (array-like): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            MinMaxDiffTransformer: Instance du transformateur.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Calcule la feature de variation minimale maximale.

        Args:
            X (array-like): Tableau de forme (n_samples, n_features)
                représentant un signal aplati (ex. image flatten).

        Returns:
            np.ndarray: Tableau de forme (n_samples, 1) contenant
            la feature extraite.
        """
        X = np.asarray(X)

        max_diffs = np.array([
            np.diff(X[:, start:end], axis=1).max(axis=1)
            for start, end in self.max_diff_ranges
        ])

        min_max_diff = max_diffs.min(axis=0).reshape(-1, 1)
        return min_max_diff
