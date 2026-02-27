import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from .transforms import get_main_contours
from .utils import are_parallel


class ParallelogramCounter(BaseEstimator, TransformerMixin):
    """
    Compte les parallélogrammes détectés dans une image.

    Les parallélogrammes sont regroupés par intervalles de ratio largeur/longueur.

    Args:
        delta_dist_tol (float): Tolérance d’approximation des contours.
        min_perimeter (float): Périmètre minimal.
        bins (list[float]): Intervalles de ratios.
    """

    def __init__(self, delta_dist_tol=0.02, min_perimeter=150,
                 bins=[0, 0.3, 0.6, 0.8, 1.1]):
        self.delta_dist_tol = delta_dist_tol
        self.min_perimeter = min_perimeter
        self.bins = bins

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        contours = get_main_contours(
            X,
            min_len=self.min_perimeter,
            min_vertex=4,
            max_vertex=4,
            delta_dist_tol=self.delta_dist_tol,
            n_max_contours=None
        )

        counter = np.zeros((len(X), len(self.bins) - 1))

        for i, conts in enumerate(contours):
            for cont in conts:
                arc = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, self.delta_dist_tol * arc, True)

                l1 = (approx[0][0], approx[1][0])
                l2 = (approx[1][0], approx[2][0])
                l3 = (approx[2][0], approx[3][0])
                l4 = (approx[3][0], approx[0][0])

                if are_parallel(l1, l3) and are_parallel(l2, l4):
                    d1 = np.linalg.norm(l1[1] - l1[0])
                    d2 = np.linalg.norm(l2[1] - l2[0])
                    L, l = max(d1, d2), min(d1, d2)
                    ratio = l / L

                    for j, (a, b) in enumerate(zip(self.bins[:-1], self.bins[1:])):
                        if a <= ratio < b:
                            counter[i, j] += 1

        return counter
