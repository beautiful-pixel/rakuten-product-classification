import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import get_squared_shape, to_square_shape, mask_background
from .contours import get_main_contours


class Flattener(BaseEstimator, TransformerMixin):
    """
    Aplati des images en vecteurs ou reconstruit des images carrées.

    Args:
        inverse_transform (bool): Si True, reconstruit les images à partir
            de vecteurs aplatis.
    """

    def __init__(self, inverse_transform: bool = False):
        self.inverse_transform = inverse_transform

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.inverse_transform:
            shape = get_squared_shape(X)
            return X.reshape(-1, *shape)
        return X.reshape(len(X), -1)

class Resizer(BaseEstimator, TransformerMixin):
    """
    Redimensionne les images et permet une conversion en niveaux de gris.

    Args:
        dsize (tuple[int, int]): Taille cible (hauteur, largeur).
        grayscale (bool): Si True, convertit les images en niveaux de gris.
    """

    def __init__(self, dsize=(64, 64), grayscale: bool = False):
        self.dsize = dsize
        self.grayscale = grayscale

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.grayscale and X.ndim == 4:
            X = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in X])
        return np.array([cv2.resize(im, self.dsize) for im in X])
    
def crop(images, contours, keep_proportion=True, padding=10):
    """
    permet de recarder des images sur des objets en fonctions de leurs contours

    """
    im_size = images.shape[1]
    cropped_images = []
    for im, conts in zip(images, contours):
        r_min, c_min = im_size, im_size
        r_max, c_max = 0, 0
        for cont in conts:
            for point in cont:
                r, c = point[0]
                if r < r_min: r_min = r
                if r > r_max: r_max = r
                if c < c_min: c_min = c
                if c > c_max: c_max = c
        if r_min >= r_max or c_min >= c_max:
            cropped_images.append(im)
        else:
            if keep_proportion:
                r_min, r_max, c_min, c_max = to_square_shape(
                    r_min, r_max, c_min, c_max, im_size, padding
                )
            if len(images.shape) == 4:
                cropped_images.append(
                    cv2.resize(im[r_min : r_max, c_min : c_max, :], dsize=(im_size, im_size))
                )
            else:
                cropped_images.append(
                    cv2.resize(im[r_min : r_max, c_min : c_max], dsize=(im_size, im_size))
                )
    return np.array(cropped_images)

def bg_crop(
    images,
    threshold: int,
    sup: bool = True,
    keep_proportion: bool = True,
    padding: int = 10,
):
    """
    Recadre les images en supprimant l’arrière-plan selon un seuil.

    Args:
        images (np.ndarray): Images d’entrée.
        threshold (int): Seuil de détection du fond.
        sup (bool): Si True, supprime les pixels supérieurs au seuil.
        keep_proportion (bool): Conserve un recadrage carré.
        padding (int): Marge ajoutée autour de la zone d’intérêt.

    Returns:
        np.ndarray: Images recadrées.
    """

    if len(images.shape) == 4:
        gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images])
    else:
        gray_images = images
    im_size = gray_images.shape[1]
    # on fait un mask qui représente les pixels d'interets (i.e. ceux qu'il ne faut pas couper) 
    if sup:
        poi = gray_images <= threshold
    else:
        poi = gray_images >= threshold

    cropped_images = []
    for i, im in enumerate(images):
        # pour chaque axe de l'image on récupère les index des vecteurs (ligne ou colonne)
        # d'interêt (la ou il y a un pixel ne doit pas être coupé)
        row_idx = np.where(poi[i].sum(axis=1) != 0)[0]
        col_idx = np.where(poi[i].sum(axis=0) != 0)[0]
        # si on ne trouve pas de région d'interêt on renvoi l'image originale
        if len(row_idx) < 2 or len(col_idx) < 2:
            cropped_images.append(im)
        # sinon on retourne l'image reacdrée
        else:
            r_min, r_max, c_min, c_max = row_idx[0], row_idx[-1], col_idx[0], col_idx[-1]
            # si on garde les proportions on doit avoir une région d'interêt carrée
            if keep_proportion:
                r_min, r_max, c_min, c_max = to_square_shape(
                    r_min, r_max, c_min, c_max, im_size, padding
                    )
            if len(images.shape) == 4:
                cropped_images.append(
                    cv2.resize(im[r_min : r_max, c_min : c_max, :], dsize=(im_size, im_size))
                    )
            else:
                cropped_images.append(
                    cv2.resize(im[r_min : r_max, c_min : c_max], dsize=(im_size, im_size))
                    )
    return np.array(cropped_images)

class ImageCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoie les images en uniformisant le fond et en recadrant sur les contours.
    """

    def __init__(
        self,
        keep_proportion: bool = True,
        padding: int = 10,
        max_bg_ratio: float = 0.7,
        white_bg: bool = True,
        min_len_ratio: float = 0.2,
        min_vertex: int = 3,
        n_max_contours: int = 7,
    ):
        self.keep_proportion = keep_proportion
        self.padding = padding
        self.max_bg_ratio = max_bg_ratio
        self.white_bg = white_bg
        self.min_len_ratio = min_len_ratio
        self.min_vertex = min_vertex
        self.n_max_contours = n_max_contours
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        min_len = self.min_len_ratio*X.shape[1]
        contours = get_main_contours(
            X, min_len=min_len, min_vertex=self.min_vertex,
            n_max_contours=self.n_max_contours
            )
        masked_ims = []
        for im, conts in zip(X, contours):
            masked_ims.append(mask_background(im, conts, self.max_bg_ratio, self.white_bg))
        masked_ims = np.array(masked_ims)
        new_X = crop(masked_ims, contours, self.keep_proportion, self.padding)
        return new_X
    
class CropTransformer(BaseEstimator, TransformerMixin):
    """
    Recadre automatiquement les images selon les zones de fond clair et sombre.
    """

    def __init__(self, padding: int = 10):
        self.padding = padding

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        # on recadre les images par rapport aux zones blanches
        X = bg_crop(X, 240, padding=self.padding)
        # puis au zones noires
        X = bg_crop(X, 15, sup=False, padding=self.padding)
        return X


class ProportionTransformer(BaseEstimator, TransformerMixin):
    """
    Calcule des proportions de catégories par échantillon.

    Pour chaque ligne de la matrice d'entrée, ce transformateur calcule
    la proportion d'éléments appartenant à chaque catégorie spécifiée.
    Les proportions sont normalisées par le nombre total d'éléments
    appartenant aux catégories considérées.
    """

    def __init__(self, categories=None):
        """
        Initialise le transformateur.

        Args:
            categories (array-like, optional): Liste des catégories à compter.
                Si None, les catégories sont déduites automatiquement à partir
                des données lors de l'appel à `fit`.
        """
        self.categories = categories

    def fit(self, X, y=None):
        """
        Ajuste le transformateur en identifiant les catégories si nécessaire.

        Args:
            X (array-like): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            ProportionTransformer: Instance du transformateur.
        """
        self.fitted_ = True
        if self.categories is None:
            self.categories = np.unique(X)
        return self

    def transform(self, X):
        """
        Calcule les proportions par catégorie pour chaque échantillon.

        Args:
            X (array-like): Tableau de forme (n_samples, n_elements)
                contenant des valeurs catégorielles ou encodées.

        Returns:
            np.ndarray: Tableau de forme (n_samples, n_categories)
            contenant les proportions par catégorie.
        """
        X = np.asarray(X)

        counts = np.array([
            (X == cat).sum(axis=1) for cat in self.categories
        ]).T

        totals = counts.sum(axis=1, keepdims=True)

        proportions = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals > 0
        )

        return proportions
