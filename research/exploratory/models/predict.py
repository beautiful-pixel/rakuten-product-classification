import torch
from tqdm import tqdm
import numpy as np

def predict_logits(model, dataloader, device):
    model.eval()
    y_true = []
    y_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Prediction"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs["logits"]
            y_logits.append(logits.cpu())
            y_true.extend(batch["labels"].cpu().tolist())

    y_logits = torch.cat(y_logits, dim=0)
    return np.array(y_true), y_logits.numpy()



def predict_proba(model, dataloader, device):
    """
    Génère les probabilités de prédiction d'un modèle de classification.

    Cette fonction est destinée à être utilisée pour :
    - l'évaluation détaillée sur le jeu de test
    - les stratégies de blending ou de stacking
    - l'analyse de calibration des probabilités

    Le modèle est automatiquement placé en mode évaluation et
    l'inférence est réalisée sans calcul de gradient.

    Args:
        model (torch.nn.Module):
            Modèle de classification entraîné.
        dataloader (torch.utils.data.DataLoader):
            Dataloader contenant les données d'entrée.
        device (str):
            Device utilisé pour l'inférence ("cuda" ou "cpu").

    Returns:
        tuple:
            - y_true (list[int]):
                Liste des labels réels.
            - y_proba (torch.Tensor):
                Tensor de forme (n_samples, n_classes) contenant
                les probabilités prédites.
    """
    model.eval()
    y_true = []
    y_proba = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Prediction"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)

            y_proba.append(probs.cpu())
            y_true.extend(batch["labels"].cpu().tolist())

    y_proba = torch.cat(y_proba, dim=0)
    return y_true, y_proba


def predict(model, dataloader, device):
    """
    Génère les prédictions de classes d'un modèle de classification.

    Cette fonction repose sur `predict_proba` et applique un argmax
    sur les probabilités prédites afin d'obtenir la classe la plus
    probable pour chaque observation.

    Args:
        model (torch.nn.Module):
            Modèle de classification entraîné.
        dataloader (torch.utils.data.DataLoader):
            Dataloader contenant les données d'entrée.
        device (str):
            Device utilisé pour l'inférence ("cuda" ou "cpu").

    Returns:
        tuple:
            - y_true (list[int]):
                Liste des labels réels.
            - y_pred (list[int]):
                Liste des classes prédites.
    """
    y_true, y_proba = predict_proba(model, dataloader, device)
    y_pred = torch.argmax(y_proba, dim=1).tolist()
    return y_true, y_pred
