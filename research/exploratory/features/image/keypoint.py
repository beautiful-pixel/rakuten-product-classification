import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans


def get_SIFT_descriptors(X):
    """
    Extrait les descripteurs SIFT d’un ensemble d’images.

    Args:
        X (np.ndarray): Images RGB.

    Returns:
        list[np.ndarray]: Liste des descripteurs par image.
    """
    sift = cv2.SIFT_create()
    descriptors = []

    for im in X:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        _, des = sift.detectAndCompute(gray, None)
        descriptors.append(des)

    return descriptors


class CornerCounter(BaseEstimator, TransformerMixin):
    """
    Compte les coins détectés par l’algorithme de Harris.

    Args:
        block_size (int): Taille du voisinage.
        ksize (int): Taille du noyau Sobel.
        threshold (float): Seuil de détection.
        apply_dilate (bool): Applique une dilatation.
    """

    def __init__(self, block_size=2, ksize=3, threshold=0.01, apply_dilate=True):
        self.block_size = block_size
        self.ksize = ksize
        self.threshold = threshold
        self.apply_dilate = apply_dilate

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if X.ndim == 4:
            X = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in X])

        counts = []
        for im in X:
            im = im.astype("float32")
            dst = cv2.cornerHarris(im, self.block_size, self.ksize, 0.05)
            if self.apply_dilate:
                dst = cv2.dilate(dst, None)
            counts.append([np.sum(dst > self.threshold * dst.max())])

        return np.array(counts)


class BoVWTransformer(BaseEstimator, TransformerMixin):
    """
    Bag of Visual Words basé sur les descripteurs SIFT.

    Args:
        n_clusters (int): Nombre de clusters visuels.
        random_state (int): Graine aléatoire.
    """

    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state
        )

    def fit(self, X, y=None):
        descriptors = [d for d in get_SIFT_descriptors(X) if d is not None]
        if len(descriptors) == 0:
            raise ValueError("Aucun descripteur SIFT trouvé.")
        self.kmeans.fit(np.vstack(descriptors))
        self.fitted_ = True
        return self

    def transform(self, X):
        descriptors = get_SIFT_descriptors(X)
        histograms = np.zeros((len(X), self.n_clusters))

        for i, des in enumerate(descriptors):
            if des is not None:
                words = self.kmeans.predict(des)
                hist, _ = np.histogram(words, bins=np.arange(self.n_clusters + 1))
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                histograms[i] = hist

        return histograms
