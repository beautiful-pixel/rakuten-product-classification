import time
from typing import Dict

from sklearn.metrics import f1_score


def evaluate_pipeline(
    pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
) -> Dict[str, float]:
    """
    Entraîne un pipeline sklearn et évalue ses performances.

    Cette fonction entraîne le pipeline sur le jeu d'entraînement,
    puis calcule les scores F1 pondérés sur les jeux
    d'entraînement et de validation. Elle mesure également
    les temps d'entraînement et de prédiction.

    Args:
        pipeline: Pipeline sklearn à évaluer.
        X_train: Features du jeu d'entraînement.
        y_train: Labels du jeu d'entraînement.
        X_val: Features du jeu de validation.
        y_val: Labels du jeu de validation.

    Returns:
        dict: Dictionnaire contenant :
            - f1_train (float): F1-score pondéré sur le train.
            - f1_val (float): F1-score pondéré sur la validation.
            - train_time_s (float): Temps d'entraînement en secondes.
            - pred_time_s (float): Temps de prédiction en secondes
              (train + validation).
    """
    # Entraînement
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prédictions
    start_time = time.time()
    y_pred_train = pipeline.predict(X_train)
    y_pred_val = pipeline.predict(X_val)
    pred_time = time.time() - start_time

    return {
        "f1_train": f1_score(
            y_train, y_pred_train, average="weighted", zero_division=0
        ),
        "f1_val": f1_score(
            y_val, y_pred_val, average="weighted", zero_division=0
        ),
        "train_time_s": train_time,
        "pred_time_s": pred_time,
    }
