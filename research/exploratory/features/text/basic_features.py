"""
Ce module fournit des fonctions pour extraire des caractéristiques statistiques
des textes de produits (designation et description) qui peuvent être utilisées
en complément des features vectorisées (TF-IDF, Count).
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def get_feature_names(text_columns=['designation', 'description']):
    features = []
    for col in text_columns:
        # Longueur
        features.extend([
            f'{col}_len_char',
            f'{col}_len_words',
        ])
        # Composition
        features.extend([
            f'{col}_num_digits',
            f'{col}_num_punctuation',
            f'{col}_uppercase_ratio',
            f'{col}_digit_ratio',
        ])
    return features



def get_length_features(series: pd.Series, prefix: str) -> pd.DataFrame:

    df = pd.DataFrame(index=series.index)

    series_filled = series.fillna('')

    # Longueur en caractères
    df[f'{prefix}_len_char'] = series_filled.str.len()

    # Longueur en mots
    df[f'{prefix}_len_words'] = series_filled.str.split().str.len()

    return df


def get_composition_features(series: pd.Series, prefix: str) -> pd.DataFrame:

    df = pd.DataFrame(index=series.index)

    series_filled = series.fillna('')

    # Nombre de chiffres
    df[f'{prefix}_num_digits'] = series_filled.str.count(r'\d')

    # Nombre de signes de ponctuation
    df[f'{prefix}_num_punctuation'] = series_filled.str.count(r'[^\w\s]')

    # Ratio de majuscules (éviter division par zéro)
    def uppercase_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)

    df[f'{prefix}_uppercase_ratio'] = series_filled.apply(uppercase_ratio)

    # Ratio de chiffres
    def digit_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isdigit()) / len(text)

    df[f'{prefix}_digit_ratio'] = series_filled.apply(digit_ratio)

    return df


def extract_text_features(
    df: pd.DataFrame,
    text_columns: List[str] = ['designation', 'description'],
    verbose: bool = False
) -> pd.DataFrame:
    if verbose:
        print("Extraction des features manuelles...")
        print(f"  Colonnes à analyser: {text_columns}")

    all_features = []

    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' introuvable dans le DataFrame")

        if verbose:
            print(f"  → Traitement de '{col}'...")

        # Extraire features de longueur
        length_feats = get_length_features(df[col], col)
        all_features.append(length_feats)

        # Extraire features de composition
        comp_feats = get_composition_features(df[col], col)
        all_features.append(comp_feats)

    # Concaténer toutes les features
    result = pd.concat(all_features, axis=1)

    if verbose:
        print(f"✓ Extraction terminée: {result.shape[1]} features créées")
        print(f"  Noms: {list(result.columns)[:3]}...")

    return result


def add_text_features(
    df: pd.DataFrame,
    text_columns: List[str] = ['designation', 'description'],
    verbose: bool = False
) -> pd.DataFrame:

    features = extract_text_features(df, text_columns, verbose)

    # Ajouter les features au DataFrame
    for col in features.columns:
        df[col] = features[col]

    return df



def describe_features(features_df: pd.DataFrame) -> pd.DataFrame:

    return features_df.describe().T


def print_feature_summary(
    df: pd.DataFrame,
    text_columns: List[str] = ['designation', 'description']
) -> None:

    feature_names = get_feature_names(text_columns)

    print("=" * 80)
    print("RÉSUMÉ DES FEATURES MANUELLES")
    print("=" * 80)

    for col in text_columns:
        print(f"\n{col.upper()}:")
        print("-" * 40)

        # Longueur
        if f'{col}_len_char' in df.columns:
            mean_chars = df[f'{col}_len_char'].mean()
            mean_words = df[f'{col}_len_words'].mean()
            print(f"  Longueur moyenne: {mean_chars:.1f} caractères, {mean_words:.1f} mots")

        # Composition
        if f'{col}_num_digits' in df.columns:
            mean_digits = df[f'{col}_num_digits'].mean()
            mean_punct = df[f'{col}_num_punctuation'].mean()
            print(f"  Composition: {mean_digits:.1f} chiffres, {mean_punct:.1f} ponctuation")

        # Ratios
        if f'{col}_uppercase_ratio' in df.columns:
            mean_upper = df[f'{col}_uppercase_ratio'].mean() * 100
            mean_digit_ratio = df[f'{col}_digit_ratio'].mean() * 100
            print(f"  Ratios: {mean_upper:.1f}% majuscules, {mean_digit_ratio:.1f}% chiffres")

    print("\n" + "=" * 80)
