from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred):
    """
    Calcule les métriques de classification.

    Args:
        y_true (list[int]): Labels réels.
        y_pred (list[int]): Prédictions.

    Returns:
        dict: Accuracy et F1 weighted.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
