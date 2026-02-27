from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from features import build_image_features_pipeline


def build_image_lr_pipeline():
    """
    Pipeline complet image + régression logistique.
    """
    features = build_image_features_pipeline()

    return Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(
            max_iter=12000,
            C=0.001,
            class_weight='balanced'
        ))
    ])
