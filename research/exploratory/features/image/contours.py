import cv2
import numpy as np

def get_length_and_vertex(contours, delta_dist_tol=0.02):
    """
    Calcule la longueur et le nombre de sommets des contours.

    Args:
        contours (list): Liste de contours OpenCV.
        delta_dist_tol (float): Tolérance d’approximation.

    Returns:
        tuple: Longueurs et nombre de sommets.
    """
    arc_len, n_vertex = [], []
    for contour in contours:
        l = cv2.arcLength(contour, True)
        arc_len.append(l)
        n_vertex.append(len(cv2.approxPolyDP(contour, delta_dist_tol * l, True)))
    return np.array(arc_len), np.array(n_vertex)


def get_main_contours(
    images,
    min_len: float = 100,
    min_vertex: int = 3,
    max_vertex: int | None = None,
    delta_dist_tol: float = 0.02,
    n_max_contours: int | None = None,
):
    """
    Détecte et filtre les contours principaux des images.

    Args:
        images (np.ndarray): Images d’entrée.
        min_len (float): Longueur minimale du contour.
        min_vertex (int): Nombre minimal de sommets.
        max_vertex (int, optional): Nombre maximal de sommets.
        delta_dist_tol (float): Tolérance pour l’approximation polygonale.
        n_max_contours (int, optional): Nombre maximal de contours conservés.

    Returns:
        list[list]: Liste de contours filtrés pour chaque image.
    """

    main_contours = []
    for im in images:
        blur = cv2.GaussianBlur(im, (5, 5), 0)
        if im.ndim == 3:
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        else:
            gray = blur
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = cv2.Canny(blur, t * 0.5, t)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.dilate(edge, kernel)

        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            main_contours.append([])
            continue

        parents = hierarchy[0][:, 3]
        parent_contours = [c for i, c in enumerate(contours) if parents[i] == -1]
        arc_len, n_vertex = get_length_and_vertex(parent_contours, delta_dist_tol)

        mask = (arc_len > min_len) & (n_vertex >= min_vertex)
        if max_vertex:
            mask &= n_vertex <= max_vertex

        filtered = [c for i, c in enumerate(parent_contours) if mask[i]]
        main_contours.append(filtered if not n_max_contours or len(filtered) <= n_max_contours else [])

    return main_contours