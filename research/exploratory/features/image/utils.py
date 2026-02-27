import cv2
import numpy as np


def get_squared_shape(X):
    """
    Détermine la forme carrée originale d’une image aplatie.

    Cette fonction permet de retrouver la forme originale (carrée)
    d’une image à partir d’un vecteur 1D représentant une image
    en niveaux de gris ou en couleur.

    Args:
        X (np.ndarray): Tableau de forme (n_samples, n_features).

    Returns:
        tuple: Forme de l’image (H, W) ou (H, W, 3).

    Raises:
        ValueError: Si le vecteur ne correspond pas à une image carrée.
    """
    if np.sqrt(X.shape[1] / 3).is_integer():
        side = int(np.sqrt(X.shape[1] / 3))
        return (side, side, 3)
    elif np.sqrt(X.shape[1]).is_integer():
        side = int(np.sqrt(X.shape[1]))
        return (side, side)
    else:
        raise ValueError("Le vecteur image doit représenter une image carrée.")


def to_square_shape(r_min, r_max, c_min, c_max, im_size, padding=10):
    """
    Calcule les coordonnées permettant un recadrage carré.

    Args:
        r_min (int): Ligne minimale.
        r_max (int): Ligne maximale.
        c_min (int): Colonne minimale.
        c_max (int): Colonne maximale.
        im_size (int): Taille de l’image carrée originale.
        padding (int): Marge ajoutée autour de l’objet.

    Returns:
        tuple: Coordonnées (r_min, r_max, c_min, c_max).
    """
    h = r_max - r_min
    w = c_max - c_min
    padding = min((im_size - max(h, w)) // 2, padding)
    crop_size = max(h, w) + 2 * padding

    if h < crop_size - 2 * padding:
        r_max += (crop_size - h) // 2
        r_min = r_max - crop_size
        c_min -= padding
        c_max += padding

    if w < crop_size - 2 * padding:
        c_max += (crop_size - w) // 2
        c_min = c_max - crop_size
        r_min -= padding
        r_max += padding

    r_min, r_max = max(0, r_min), min(im_size, r_max)
    c_min, c_max = max(0, c_min), min(im_size, c_max)

    return r_min, r_max, c_min, c_max


def are_parallel(line1, line2, tol=0.03):
    """
    Vérifie si deux segments sont parallèles.

    Args:
        line1 (tuple): Segment ((x1, y1), (x2, y2)).
        line2 (tuple): Segment ((x3, y3), (x4, y4)).
        tol (float): Tolérance angulaire (en fraction de π).

    Returns:
        bool: True si les segments sont parallèles.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    u = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    v = np.array([x4 - x3, y4 - y3], dtype=np.float32)

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    theta = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    return theta < np.pi * tol or theta > np.pi * (1 - tol)

def mask_background(
    image,
    contours,
    max_bg_ratio: float = 0.7,
    white_bg: bool = True,
):
    """
    Masque l’arrière-plan d’une image à partir de contours.

    Args:
        image (np.ndarray): Image d’entrée.
        contours (list): Contours à conserver.
        max_bg_ratio (float): Ratio maximal de fond autorisé.
        white_bg (bool): Remplace le fond par du blanc si True.

    Returns:
        np.ndarray: Image avec arrière-plan masqué.
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    background_ratio = (mask==0).mean()
    if background_ratio > max_bg_ratio:
        masked_image = image.copy()
    else:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        if white_bg:
            masked_image[mask == 0] = 255
    return masked_image