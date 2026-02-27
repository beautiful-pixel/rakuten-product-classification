from typing import List, Optional, Dict, Tuple, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Configurations par Défaut

DEFAULT_CONFIG = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'lowercase': False,  # Déjà fait dans preprocessing
    'strip_accents': None,  # Déjà fait dans preprocessing
}

# Configurations prédéfinies pour tests rapides
PRESET_CONFIGS = {
    'baseline': {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
    },
    'small': {
        'max_features': 5000,
        'ngram_range': (1, 1),
        'min_df': 2,
        'max_df': 0.95,
    },
    'large': {
        'max_features': 20000,
        'ngram_range': (1, 3),
        'min_df': 2,
        'max_df': 0.95,
    },
    'unigram': {
        'max_features': 10000,
        'ngram_range': (1, 1),
        'min_df': 2,
        'max_df': 0.95,
    },
    'trigram': {
        'max_features': 15000,
        'ngram_range': (1, 3),
        'min_df': 2,
        'max_df': 0.95,
    }
}


# Constructeurs de Vectoriseurs

def build_count_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    **kwargs
) -> CountVectorizer:

    config = {
        'max_features': max_features,
        'ngram_range': ngram_range,
        'min_df': min_df,
        'max_df': max_df,
        'lowercase': False,  # Assumé déjà fait
        'strip_accents': None,
        **kwargs
    }

    return CountVectorizer(**config)


def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    **kwargs
) -> TfidfVectorizer:
    config = {
        'max_features': max_features,
        'ngram_range': ngram_range,
        'min_df': min_df,
        'max_df': max_df,
        'sublinear_tf': sublinear_tf,
        'lowercase': False,
        'strip_accents': None,
        **kwargs
    }

    return TfidfVectorizer(**config)


def build_vectorizer(
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    preset: Optional[str] = None,
    **kwargs
) -> CountVectorizer | TfidfVectorizer:
    
    # Charger le preset si fourni
    config = {}
    if preset:
        if preset not in PRESET_CONFIGS:
            raise ValueError(
                f"Preset '{preset}' inconnu. "
                f"Disponibles: {list(PRESET_CONFIGS.keys())}"
            )
        config = PRESET_CONFIGS[preset].copy()

    # Fusionner avec kwargs (kwargs prioritaires)
    config.update(kwargs)

    # Créer le vectoriseur approprié
    if vectorizer_type == 'count':
        return build_count_vectorizer(**config)
    elif vectorizer_type == 'tfidf':
        return build_tfidf_vectorizer(**config)
    else:
        raise ValueError(
            f"vectorizer_type '{vectorizer_type}' invalide. "
            f"Choisir 'count' ou 'tfidf'."
        )


# Custom Transformers

class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Transformer personnalisé pour pondérer les features par un facteur multiplicatif.

    Utilisé pour augmenter l'importance relative de certaines features
    (ex: multiplier les features du titre par 2x ou 3x).

    Hérite de BaseEstimator et TransformerMixin pour être compatible avec
    sklearn et picklable pour le multiprocessing.
    """

    def __init__(self, weight: float = 1.0):
        """
        Paramètres
        ----------
        weight : float, default=1.0
            Facteur multiplicatif à appliquer aux features
        """
        self.weight = weight

    def fit(self, X, y=None):
        """Fit method (no-op)."""
        return self

    def transform(self, X):
        """Multiplie les features par le poids."""
        return X * self.weight


# Pipelines de Vectorisation

def build_split_vectorizer_pipeline(
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    text_columns: List[str] = ['title_clean', 'desc_clean'],
    feature_columns: Optional[List[str]] = None,
    max_features_title: int = 10000,
    max_features_desc: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    title_weight: float = 1.0,
    **kwargs
) -> ColumnTransformer:
    
    if len(text_columns) < 2:
        raise ValueError("split_vectorizer_pipeline requiert au moins 2 colonnes texte")

    transformers = []

    # Vectoriseur pour le titre
    vec_title = build_vectorizer(
        vectorizer_type,
        max_features=max_features_title,
        ngram_range=ngram_range,
        **kwargs
    )

    # Appliquer la pondération si différente de 1.0
    if title_weight != 1.0:
        title_pipeline = Pipeline([
            ('vectorizer', vec_title),
            ('weighter', FeatureWeighter(weight=title_weight))
        ])
        transformers.append(('vec_title', title_pipeline, text_columns[0]))
    else:
        transformers.append(('vec_title', vec_title, text_columns[0]))

    # Vectoriseur pour la description
    vec_desc = build_vectorizer(
        vectorizer_type,
        max_features=max_features_desc,
        ngram_range=ngram_range,
        **kwargs
    )
    transformers.append(('vec_desc', vec_desc, text_columns[1]))

    # Features manuelles (si fournies)
    if feature_columns:
        transformers.append(('scaler', StandardScaler(), feature_columns))

    # Désactiver parallélisation si title_weight != 1.0 pour éviter erreurs de sérialisation
    n_jobs = 1 if title_weight != 1.0 else -1

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        n_jobs=n_jobs
    )


def build_merged_vectorizer_pipeline(
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    text_column: str = 'text_merged',
    feature_columns: Optional[List[str]] = None,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    **kwargs
) -> ColumnTransformer:
    
    transformers = []

    # Vectoriseur pour le texte fusionné
    vec = build_vectorizer(
        vectorizer_type,
        max_features=max_features,
        ngram_range=ngram_range,
        **kwargs
    )
    transformers.append(('vectorizer', vec, text_column))

    # Features manuelles (si fournies)
    if feature_columns:
        transformers.append(('scaler', StandardScaler(), feature_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        n_jobs=-1
    )


# =============================================================================
# Utilitaires
# =============================================================================

def get_available_presets() -> Dict[str, Dict]:
    
    return PRESET_CONFIGS.copy()


def print_available_presets() -> None:
    
    print("Configurations prédéfinies disponibles:")
    print("=" * 80)

    for name, config in PRESET_CONFIGS.items():
        print(f"\n  {name}:")
        for key, value in config.items():
            print(f"    {key}: {value}")

    print("\n" + "=" * 80)


def get_vectorizer_info(vectorizer) -> Dict[str, any]:
    
    return {
        'type': type(vectorizer).__name__,
        'max_features': vectorizer.max_features,
        'ngram_range': vectorizer.ngram_range,
        'min_df': vectorizer.min_df,
        'max_df': vectorizer.max_df,
    }


def compare_vectorizer_configs(vec1, vec2) -> None:
    
    info1 = get_vectorizer_info(vec1)
    info2 = get_vectorizer_info(vec2)

    print("Comparaison de configurations:")
    print("=" * 80)
    print(f"{'Paramètre':<20} {'Vectorizer 1':<30} {'Vectorizer 2':<30}")
    print("-" * 80)

    all_keys = set(info1.keys()) | set(info2.keys())
    for key in sorted(all_keys):
        val1 = info1.get(key, 'N/A')
        val2 = info2.get(key, 'N/A')
        print(f"{key:<20} {str(val1):<30} {str(val2):<30}")

    print("=" * 80)


# Configuration Management (pour pipeline entre notebooks)

def save_vectorization_config(
    config: Dict[str, any],
    output_path: str,
    metadata: Optional[Dict[str, any]] = None
) -> None:
    
    import json
    from datetime import datetime

    # Créer le dictionnaire complet
    full_config = {
        **config,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }

    # Convertir tuples en listes pour JSON
    if 'ngram_range' in full_config and isinstance(full_config['ngram_range'], tuple):
        full_config['ngram_range'] = list(full_config['ngram_range'])

    # Créer le répertoire si nécessaire
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(full_config, f, indent=2)

    print(f"✓ Configuration sauvegardée: {output_path}")


def load_vectorization_config(config_path: str) -> ColumnTransformer:
    
    import json

    # Charger la configuration
    with open(config_path) as f:
        config = json.load(f)

    print(f"Configuration chargée depuis: {config_path}")
    print(f"  Vectorizer: {config.get('vectorizer_type', 'N/A')}")
    print(f"  Strategy: {config.get('strategy', 'N/A')}")

    # Convertir listes en tuples
    if 'ngram_range' in config and isinstance(config['ngram_range'], list):
        config['ngram_range'] = tuple(config['ngram_range'])

    # Construire le vectorizer selon la stratégie
    if config.get('strategy') == 'split':
        vectorizer = build_split_vectorizer_pipeline(
            vectorizer_type=config['vectorizer_type'],
            text_columns=config.get('text_columns', ['title_clean', 'desc_clean']),
            feature_columns=config.get('feature_columns'),
            max_features_title=config.get('max_features_title', 10000),
            max_features_desc=config.get('max_features_desc', 10000),
            ngram_range=config.get('ngram_range', (1, 2)),
            title_weight=config.get('title_weight', 1.0),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.95)
        )
    else:  # merged
        vectorizer = build_merged_vectorizer_pipeline(
            vectorizer_type=config['vectorizer_type'],
            text_column=config.get('text_column', 'text_merged'),
            feature_columns=config.get('feature_columns'),
            max_features=config.get('max_features', 10000),
            ngram_range=config.get('ngram_range', (1, 2)),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.95)
        )

    print("✓ Vectorizer construit avec succès")
    return vectorizer


def get_config_summary(config_path: str) -> None:
    
    import json

    with open(config_path) as f:
        config = json.load(f)

    print("=" * 80)
    print("RÉSUMÉ DE LA CONFIGURATION")
    print("=" * 80)
    print(f"\nVectorizer: {config.get('vectorizer_type', 'N/A').upper()}")
    print(f"Strategy: {config.get('strategy', 'N/A').capitalize()}")
    print(f"N-gram range: {config.get('ngram_range', 'N/A')}")

    if config.get('strategy') == 'split':
        print(f"Max features (title): {config.get('max_features_title', 'N/A')}")
        print(f"Max features (desc): {config.get('max_features_desc', 'N/A')}")
    else:
        print(f"Max features: {config.get('max_features', 'N/A')}")

    print(f"Min df: {config.get('min_df', 'N/A')}")
    print(f"Max df: {config.get('max_df', 'N/A')}")

    if config.get('feature_columns'):
        print(f"Features manuelles: {len(config['feature_columns'])} colonnes")
    else:
        print("Features manuelles: Non")

    if 'metadata' in config:
        print("\nMétadonnées:")
        for key, value in config['metadata'].items():
            print(f"  {key}: {value}")

    print("=" * 80)
