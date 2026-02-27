import numpy as np
import pandas as pd
import cv2
from sklearn.base import BaseEstimator, TransformerMixin


def is_pixel_in_bin(X, ranges):
    """
    Détermine si les pixels appartiennent à un intervalle HSV donné.

    Les intervalles sont semi-ouverts à droite.

    Args:
        X (np.ndarray): Images HSV de forme (n, h, w, 3).
        ranges (list): Intervalles [[Hmin, Hmax], [Smin, Smax], [Vmin, Vmax]].

    Returns:
        np.ndarray: Masque booléen des pixels appartenant à l’intervalle.
    """
    mask = {}
    for channel, r in enumerate(ranges):
        if r[0] > r[1]:
            mask[channel] = (X[:, :, :, channel] >= r[0]) | (X[:, :, :, channel] < r[1])
        else:
            mask[channel] = (X[:, :, :, channel] >= r[0]) & (X[:, :, :, channel] < r[1])
    return mask[0] & mask[1] & mask[2]


class ColorEncoder(BaseEstimator, TransformerMixin):
    """
    Encode chaque pixel selon des catégories de couleur HSV.

    Chaque pixel est associé à une classe de couleur prédéfinie.
    Une option permet de retourner un encodage one-hot par pixel.

    Args:
        ohe (bool): Si True, applique un encodage one-hot.
    """

    def __init__(self, ohe=False):
        self.ohe = ohe
        self.labels = [
            "noir", "gris", "blanc", "rouge", "orange", "jaune",
            "vert", "turquoise", "cyan", "bleu", "violet", "rose"
        ]

        ranges = [
            [[0, 181], [0, 256], [0, 70]],
            [[0, 181], [0, 40], [70, 200]],
            [[0, 181], [0, 40], [200, 256]],
        ]

        ncolors = 9
        half_step = (180 // ncolors) / 2
        for i in np.linspace(0, 180 - 2 * half_step, ncolors):
            if i == 0:
                ranges.append([[180 - half_step, half_step], [40, 256], [70, 256]])
            else:
                ranges.append([[i - half_step, i + half_step], [40, 256], [70, 256]])

        self.ranges = ranges

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2HSV) for im in X])
        encoded = np.full(X.shape[:3], -1)

        for i, r in enumerate(self.ranges):
            encoded[is_pixel_in_bin(X, r)] = i

        if self.ohe:
            encoded = np.eye(len(self.labels))[encoded].reshape(len(encoded), -1)

        return encoded
    
    def get_centroids(self, rgb=False):
        """
        retourne la liste des centroids des intervalles
        """
        centroids = []
        for r in self.ranges:
            h_min, h_max = r[0][0], r[0][1]
            s_min, s_max =  r[1][0], r[1][1]
            v_min, v_max = r[2][0], r[2][1]
            if h_min > h_max:
                h_min -= 180
            h = int(h_min + (h_max-h_min)/2)
            s = int(s_min + (s_max-s_min)/2)
            v = int(v_min + (v_max-v_min)/2)
            centroids.append([h,s,v])
        # on ajoute une dimension pour que la matrice puisse représenter une image couleur
        # et être convertie en rgb si nécessaire
        centroids = np.array(centroids, dtype='uint8').reshape(1,-1,3)
        if rgb:
            centroids = cv2.cvtColor(centroids, cv2.COLOR_HSV2RGB)
        return centroids


class MeanRGBTransformer(BaseEstimator, TransformerMixin):
    """
    Calcule la moyenne des canaux RGB d’une image.

    Args:
        channels (list[int]): Canaux RGB à utiliser.
        ignore_white_black (bool): Ignore les pixels blancs et noirs.
        impute_value (int): Valeur utilisée si aucun pixel valide.
    """

    def __init__(self, channels=[0, 1, 2], ignore_white_black=True, impute_value=127):
        self.channels = channels
        self.ignore_white_black = ignore_white_black
        self.impute_value = impute_value

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.ignore_white_black:
            masks = []
            for im in X:
                hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                s, v = hsv[:, :, 1], hsv[:, :, 2]
                masks.append((v > 70) & ((s > 40) | (v < 200)))
            masks = np.array(masks)
        else:
            masks = np.ones(X.shape[:3], dtype=bool)

        means = []
        for i in range(len(X)):
            row = []
            for c in self.channels:
                values = X[i, :, :, c][masks[i]]
                row.append(values.mean() if len(values) > 0 else self.impute_value)
            means.append(row)

        return np.array(means)


class HistRGBTransformer(BaseEstimator, TransformerMixin):
    """
    Calcule des histogrammes RGB par canal.

    Args:
        histSize (list[int]): Nombre de bins.
        channels (list[int]): Canaux RGB.
        ranges (list): Intervalles des valeurs.
    """

    def __init__(self, histSize=[256], channels=[0, 1, 2],
                 ranges=[[0, 256], [0, 256], [0, 256]]):
        self.histSize = histSize
        self.channels = channels
        self.ranges = ranges

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        hist_list = []
        col_names = []
        cols = "rgb"

        for channel in self.channels:
            base = self.ranges[channel][0]
            size = self.ranges[channel][1] - base
            col_names += [
                cols[channel] + str(int(base + i * size / self.histSize[0]))
                for i in range(self.histSize[0])
            ]

            hists = np.array([
                cv2.calcHist([im], [channel], None, self.histSize, self.ranges[channel])
                for im in X
            ])
            hist_list.append(hists)

        return pd.DataFrame(
            np.hstack(hist_list).reshape(len(X), -1),
            columns=col_names
        )
