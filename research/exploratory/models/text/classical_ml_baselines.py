import time
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from xgboost import XGBClassifier
from ...data.categories import get_category_name


def get_model(model_name, random_state=42, **kwargs):

    model_name = model_name.lower().strip()

    if model_name == 'logreg':
        # Régression Logistique multinomiale
        # Baseline solide pour la classification de texte
        # solver='lbfgs' est robuste pour les données multiclasses
        default_params = {
            'C': 1.0,                    # Régularisation inverse (plus haut = moins de régularisation)
            'solver': 'lbfgs',           # Optimiseur efficace pour multiclasse
            'max_iter': 1000,            # Nombre max d'itérations pour convergence
            'random_state': random_state,
            'n_jobs': 1,                # Utiliser tous les cœurs CPU
            'verbose': 0
        }
        default_params.update(kwargs)
        return LogisticRegression(**default_params)

    elif model_name == 'svm':
        # SVM linéaire via Stochastic Gradient Descent
        # Approximation efficace du SVM pour grands datasets
        # loss='hinge' correspond au SVM classique
        default_params = {
            'loss': 'hinge',             # Fonction de perte SVM
            'alpha': 1e-4,               # Coefficient de régularisation L2
            'max_iter': 1000,            # Nombre max d'epochs
            'tol': 1e-3,                 # Critère de convergence
            'random_state': random_state,
            'n_jobs': 1,
            'verbose': 0
        }
        default_params.update(kwargs)
        return SGDClassifier(**default_params)

    elif model_name == 'xgboost':
        # XGBoost: Gradient Boosting optimisé
        # Capture les interactions non-linéaires entre features
        # Très performant pour la classification multiclasse
        default_params = {
            'use_label_encoder': False,  # Désactivé (deprecated)
            'eval_metric': 'mlogloss',   # Log-loss multiclasse
            'n_estimators': 100,         # Nombre d'arbres (augmenter si sous-apprentissage)
            'max_depth': 6,              # Profondeur max des arbres
            'learning_rate': 0.3,        # Taux d'apprentissage
            'random_state': random_state,
            'n_jobs': 1,
            'verbosity': 0
        }
        default_params.update(kwargs)
        return XGBClassifier(**default_params)

    elif model_name == 'rf':
        # Random Forest: Ensemble de decision trees
        # Robuste au surapprentissage, parallélisable
        # Baseline non-linéaire complémentaire à LogReg
        default_params = {
            'n_estimators': 100,         # Nombre d'arbres dans la forêt
            'max_depth': None,           # Profondeur illimitée (contrôlée par min_samples_split)
            'min_samples_split': 2,      # Min échantillons pour splitter un nœud
            'min_samples_leaf': 1,       # Min échantillons dans une feuille
            'random_state': random_state,
            'n_jobs': 1,
            'verbose': 0
        }
        default_params.update(kwargs)
        return RandomForestClassifier(**default_params)

    else:
        raise ValueError(
            f"Modèle '{model_name}' non reconnu. "
            f"Choisir parmi: 'logreg', 'svm', 'xgboost', 'rf'"
        )


def get_available_models():
    return ['logreg', 'svm', 'xgboost', 'rf']


def get_model_info(model_name):

    info = {
        'logreg': "Régression Logistique: Modèle linéaire baseline, rapide et interprétable.",
        'svm': "SVM Linéaire: Maximisation de marge, efficace pour espaces haute dimension.",
        'xgboost': "XGBoost: Gradient Boosting, capture des interactions non-linéaires complexes.",
        'rf': "Random Forest: Ensemble d'arbres, robuste et parallélisable."
    }
    return info.get(model_name.lower(), "Modèle inconnu")



def build_full_pipeline(vectorizer, model_name='logreg', random_state=42, **model_kwargs):

    model = get_model(model_name, random_state=random_state, **model_kwargs)

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])

    return pipeline


def evaluate_pipeline(
    pipeline,
    X_val,
    y_val,
    verbose: bool = True,
    show_category_names: bool = True
) -> Dict[str, Any]:

    # Prédiction avec mesure du temps
    start_time = time.time()
    y_pred = pipeline.predict(X_val)
    predict_time = time.time() - start_time

    # Calcul des métriques
    f1 = f1_score(y_val, y_pred, average='weighted')
    acc = accuracy_score(y_val, y_pred)

    # Affichage verbose
    if verbose:
        print("=" * 80)
        print("RÉSULTATS DE L'ÉVALUATION")
        print("=" * 80)
        print(f"F1-Score (weighted): {f1:.4f}")
        print(f"Accuracy:            {acc:.4f}")
        print(f"Temps prédiction:    {predict_time:.2f}s")
        print("\nRapport de classification:")
        print("-" * 80)

        # Ajouter les noms de catégories si demandé
        if show_category_names:
            import numpy as np
            unique_labels = np.unique(np.concatenate([y_val, y_pred]))
            target_names = [get_category_name(int(code), short=True) for code in unique_labels]
            print(classification_report(y_val, y_pred, target_names=target_names))
        else:
            print(classification_report(y_val, y_pred))

    return {
        'f1_score': f1,
        'accuracy': acc,
        'y_pred': y_pred,
        'predict_time': predict_time
    }


def train_and_evaluate(
    pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose: bool = True,
    show_category_names: bool = True
) -> Dict[str, Any]:

    if verbose:
        print("Entraînement du pipeline...")

    # Entraînement avec mesure du temps
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    if verbose:
        print(f"✓ Entraînement terminé en {train_time:.2f}s\n")

    # Évaluation
    eval_results = evaluate_pipeline(pipeline, X_val, y_val, verbose=verbose,
                                     show_category_names=show_category_names)

    # Combiner les résultats
    results = {
        **eval_results,
        'train_time': train_time,
        'pipeline': pipeline
    }

    return results
