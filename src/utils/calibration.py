"""
Outils de calibration post-hoc des modèles de deep learning.

Ce module fournit des fonctions permettant de calibrer les sorties
(logits) de modèles de classification à l'aide du temperature scaling.
La calibration est appliquée après l'entraînement des modèles et vise
à améliorer la cohérence probabiliste des prédictions, notamment dans
le cadre de fusions de modèles (blending ou stacking).

Ces fonctions ne modifient pas les prédictions de classe (argmax),
mais uniquement la distribution des probabilités.
"""

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iters: int = 200,
    lr: float = 0.05
) -> float:
    """
    Apprend un paramètre de température pour le temperature scaling.

    Cette fonction ajuste un unique scalaire positif T permettant de
    calibrer les logits d'un modèle en minimisant la fonction de perte
    de type entropie croisée sur un jeu de validation.

    Les probabilités calibrées sont ensuite obtenues via :

        softmax(logits / T)

    Le temperature scaling est une méthode de calibration post-hoc qui
    améliore la fiabilité des probabilités prédites sans modifier les
    classes prédites par le modèle.

    Paramètres
    ----------
    logits : np.ndarray de forme (n_samples, n_classes)
        Sorties brutes du modèle avant application de la fonction softmax.
        Ces valeurs correspondent aux logits produits par la dernière
        couche linéaire du classifieur.
    labels : np.ndarray de forme (n_samples,)
        Labels réels associés aux échantillons du jeu de validation.
    max_iters : int, optionnel (par défaut = 200)
        Nombre maximal d'itérations utilisées pour optimiser la température.
    lr : float, optionnel (par défaut = 0.05)
        Taux d'apprentissage de l'optimiseur.

    Retourne
    --------
    float
        Température apprise T (T > 0).  
        - T < 1 : les probabilités deviennent plus piquées (logits amplifiés)  
        - T > 1 : les probabilités deviennent plus lisses (logits atténués)

    Notes
    -----
    - La température doit être apprise uniquement sur le jeu de validation
      et ne doit jamais être recalibrée sur le jeu de test.
    - Cette méthode n'introduit qu'un seul paramètre scalaire et présente
      donc un risque très faible de sur-apprentissage.
    - La même température doit être réutilisée pour toutes les prédictions
      futures du modèle (validation finale, test).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    log_T = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_T], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        optimizer.step()

    T = float(torch.exp(log_T).detach().cpu().item())
    return max(T, 1e-3)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """
    Applique la fonction softmax de manière numériquement stable.

    Cette implémentation soustrait la valeur maximale des logits pour
    chaque échantillon afin d'éviter les débordements numériques lors
    de l'exponentiation.

    Paramètres
    ----------
    logits : np.ndarray de forme (n_samples, n_classes)
        Tableau de logits à normaliser.

    Retourne
    --------
    np.ndarray de forme (n_samples, n_classes)
        Probabilités normalisées pour chaque classe.
    """
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def calibrated_probas(
    logits: np.ndarray,
    T: float
) -> np.ndarray:
    """
    Convertit des logits en probabilités calibrées à l'aide du temperature scaling.

    Cette fonction applique successivement le temperature scaling et la
    fonction softmax afin d'obtenir des probabilités calibrées :

        probabilités = softmax(logits / T)

    Paramètres
    ----------
    logits : np.ndarray de forme (n_samples, n_classes)
        Sorties brutes du modèle avant softmax.
    T : float
        Paramètre de température appris à l'aide de la fonction
        `fit_temperature`. Doit être strictement positif.

    Retourne
    --------
    np.ndarray de forme (n_samples, n_classes)
        Probabilités calibrées associées à chaque classe.
    """
    return softmax_np(logits / T)

# src/utils/calibrations.py

import numpy as np


def normalize_probas(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize probability distributions so that each row sums to 1.

    This function is used after blending or stacking operations to ensure
    numerical stability and valid probability distributions.

    Args:
        P (np.ndarray): Array of shape (n_samples, n_classes) containing
            unnormalized or approximately normalized probabilities.
        eps (float): Small constant to avoid numerical issues.

    Returns:
        np.ndarray: Normalized probability array with rows summing to 1.
    """
    P = np.clip(P, eps, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def weights_from_logloss(log_losses: dict) -> dict:
    """
    Compute blending weights inversely proportional to log-loss values.

    Lower log-loss values result in higher weights. The resulting weights
    are normalized to sum to 1.

    Args:
        log_losses (dict): Dictionary mapping model names to their log-loss
            values computed on a validation set.

    Returns:
        dict: Dictionary mapping model names to normalized blending weights.
    """
    inv = {k: 1.0 / v for k, v in log_losses.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}
