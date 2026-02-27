"""
Ce module fournit des fonctions pour exécuter des expériences systématiques
comparant CountVectorizer vs TF-IDF, avec et sans features manuelles,
ainsi que des grid searches pour les hyperparamètres.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Literal, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from ..features.text.basic_features import extract_text_features, get_feature_names
from ..features.text.vectorization import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
    build_split_vectorizer_pipeline,
    build_merged_vectorizer_pipeline
)
from ..models.text.classical_ml_baselines import build_full_pipeline, train_and_evaluate, get_model


# 
def run_single_experiment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    strategy: Literal['split', 'merged'] = 'split',
    text_columns: List[str] = ['title_clean', 'desc_clean'],
    feature_columns: Optional[List[str]] = None,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    model_name: str = 'logreg',
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:

    print(f"\n{'='*80}")
    print(f"Expérience: {vectorizer_type.upper()} + {model_name.upper()}")
    print(f"  Strategy: {strategy}")
    print(f"  Max features: {max_features}")
    print(f"  N-gram range: {ngram_range}")
    print(f"  Features manuelles: {'Oui' if feature_columns else 'Non'}")
    print(f"{'='*80}\n")

    # Construire le pipeline de vectorisation
    if strategy == 'split':
        vectorizer = build_split_vectorizer_pipeline(
            vectorizer_type=vectorizer_type,
            text_columns=text_columns,
            feature_columns=feature_columns,
            max_features_title=max_features,
            max_features_desc=max_features,
            ngram_range=ngram_range,
            **kwargs
        )
    else:  # merged
        # Assumer que la colonne merged existe
        merged_col = text_columns[0] if len(text_columns) == 1 else 'text_merged'
        vectorizer = build_merged_vectorizer_pipeline(
            vectorizer_type=vectorizer_type,
            text_column=merged_col,
            feature_columns=feature_columns,
            max_features=max_features,
            ngram_range=ngram_range,
            **kwargs
        )

    # Construire et entraîner le pipeline complet
    pipeline = build_full_pipeline(vectorizer, model_name=model_name)
    results = train_and_evaluate(
        pipeline, X_train, y_train, X_val, y_val, verbose=verbose
    )

    # Ajouter les métadonnées de configuration
    results['config'] = {
        'vectorizer_type': vectorizer_type,
        'strategy': strategy,
        'max_features': max_features,
        'ngram_range': ngram_range,
        'model_name': model_name,
        'use_features': feature_columns is not None,
        'n_feature_cols': len(feature_columns) if feature_columns else 0
    }

    return results


# Grid Search Hyperparamètres

def run_hyperparameter_grid(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    max_features_list: List[int] = [5000, 10000, 20000],
    ngram_range_list: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 3)],
    model_name: str = 'logreg',
    text_columns: List[str] = ['title_clean', 'desc_clean'],
    feature_columns: Optional[List[str]] = None,
    title_weight: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:

    if verbose:
        total = len(max_features_list) * len(ngram_range_list)
        print(f"\n{'='*80}")
        print(f"GRID SEARCH: {vectorizer_type.upper()} Hyperparameters")
        print(f"{'='*80}")
        print(f"Total expériences: {total}")
        print(f"  max_features: {max_features_list}")
        print(f"  ngram_range: {ngram_range_list}")
        print(f"  title_weight: {title_weight}")
        print(f"  model: {model_name}")
        print(f"{'='*80}\n")

    results = []
    exp_count = 0

    for max_feats in max_features_list:
        for ngram in ngram_range_list:
            exp_count += 1
            if verbose:
                print(f"[{exp_count}/{total}] max_features={max_feats}, ngram_range={ngram}")

            result = run_single_experiment(
                X_train, X_val, y_train, y_val,
                vectorizer_type=vectorizer_type,
                strategy='split',
                text_columns=text_columns,
                feature_columns=feature_columns,
                max_features=max_feats,
                ngram_range=ngram,
                model_name=model_name,
                title_weight=title_weight,
                verbose=False
            )

            # Simplifier pour DataFrame
            results.append({
                'vectorizer': vectorizer_type,
                'max_features': max_feats,
                'ngram_range': str(ngram),
                'title_weight': title_weight,
                'model': model_name,
                'use_features': feature_columns is not None,
                'f1_score': result['f1_score'],
                'accuracy': result['accuracy'],
                'train_time': result['train_time'],
                'predict_time': result['predict_time']
            })

            if verbose:
                print(f"  → F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}\n")

    df_results = pd.DataFrame(results).sort_values('f1_score', ascending=False)

    if verbose:
        print(f"\n{'='*80}")
        print("MEILLEURE CONFIGURATION:")
        print(f"{'='*80}")
        best = df_results.iloc[0]
        print(f"  Max features: {best['max_features']}")
        print(f"  N-gram range: {best['ngram_range']}")
        print(f"  F1-Score: {best['f1_score']:.4f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"{'='*80}\n")

    return df_results


# Comparaison de Stratégies

def run_strategy_comparison(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    text_columns: List[str] = ['title_clean', 'desc_clean'],
    feature_columns: Optional[List[str]] = None,
    models: List[str] = ['logreg'],
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    verbose: bool = True
) -> pd.DataFrame:

    if verbose:
        print(f"\n{'='*80}")
        print("COMPARAISON COMPLÈTE DES STRATÉGIES DE VECTORISATION")
        print(f"{'='*80}")
        print(f"Modèles à tester: {models}")
        print(f"Max features: {max_features}")
        print(f"N-gram range: {ngram_range}")
        print(f"{'='*80}\n")

    all_results = []

    # Stratégie 1: Indicateurs seuls (si fournis)
    if feature_columns:
        if verbose:
            print("\n📊 STRATÉGIE 1: Indicateurs seuls")
            print("-" * 80)

        for model in models:
            if verbose:
                print(f"  Modèle: {model.upper()}")

            # Créer un pipeline simple avec StandardScaler + Model
            from sklearn.pipeline import Pipeline
            scaler = StandardScaler()
            clf = get_model(model)

            pipeline = Pipeline([
                ('scaler', scaler),
                ('classifier', clf)
            ])

            # Sélectionner uniquement les feature columns
            X_train_feat = X_train[feature_columns]
            X_val_feat = X_val[feature_columns]

            start = time.time()
            pipeline.fit(X_train_feat, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = pipeline.predict(X_val_feat)
            predict_time = time.time() - start

            from sklearn.metrics import f1_score, accuracy_score
            f1 = f1_score(y_val, y_pred, average='weighted')
            acc = accuracy_score(y_val, y_pred)

            all_results.append({
                'strategy': 'Indicateurs seuls',
                'vectorizer': 'None',
                'model': model,
                'use_features': True,
                'f1_score': f1,
                'accuracy': acc,
                'train_time': train_time,
                'predict_time': predict_time
            })

            if verbose:
                print(f"    F1: {f1:.4f}, Acc: {acc:.4f}\n")

    # Stratégie 2: CountVectorizer seul
    if verbose:
        print("\n📊 STRATÉGIE 2: CountVectorizer seul")
        print("-" * 80)

    for model in models:
        if verbose:
            print(f"  Modèle: {model.upper()}")

        result = run_single_experiment(
            X_train, X_val, y_train, y_val,
            vectorizer_type='count',
            strategy='split',
            text_columns=text_columns,
            feature_columns=None,  # Pas de features
            max_features=max_features,
            ngram_range=ngram_range,
            model_name=model,
            verbose=False
        )

        all_results.append({
            'strategy': 'CountVectorizer seul',
            'vectorizer': 'count',
            'model': model,
            'use_features': False,
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'train_time': result['train_time'],
            'predict_time': result['predict_time']
        })

        if verbose:
            print(f"    F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}\n")

    # Stratégie 3: TF-IDF seul
    if verbose:
        print("\n📊 STRATÉGIE 3: TF-IDF seul")
        print("-" * 80)

    for model in models:
        if verbose:
            print(f"  Modèle: {model.upper()}")

        result = run_single_experiment(
            X_train, X_val, y_train, y_val,
            vectorizer_type='tfidf',
            strategy='split',
            text_columns=text_columns,
            feature_columns=None,
            max_features=max_features,
            ngram_range=ngram_range,
            model_name=model,
            verbose=False
        )

        all_results.append({
            'strategy': 'TF-IDF seul',
            'vectorizer': 'tfidf',
            'model': model,
            'use_features': False,
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'train_time': result['train_time'],
            'predict_time': result['predict_time']
        })

        if verbose:
            print(f"    F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}\n")

    # Stratégie 4: Indicateurs + CountVectorizer
    if feature_columns:
        if verbose:
            print("\n📊 STRATÉGIE 4: Indicateurs + CountVectorizer")
            print("-" * 80)

        for model in models:
            if verbose:
                print(f"  Modèle: {model.upper()}")

            result = run_single_experiment(
                X_train, X_val, y_train, y_val,
                vectorizer_type='count',
                strategy='split',
                text_columns=text_columns,
                feature_columns=feature_columns,
                max_features=max_features,
                ngram_range=ngram_range,
                model_name=model,
                verbose=False
            )

            all_results.append({
                'strategy': 'Indicateurs + CountVectorizer',
                'vectorizer': 'count',
                'model': model,
                'use_features': True,
                'f1_score': result['f1_score'],
                'accuracy': result['accuracy'],
                'train_time': result['train_time'],
                'predict_time': result['predict_time']
            })

            if verbose:
                print(f"    F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}\n")

    # Stratégie 5: Indicateurs + TF-IDF
    if feature_columns:
        if verbose:
            print("\n📊 STRATÉGIE 5: Indicateurs + TF-IDF")
            print("-" * 80)

        for model in models:
            if verbose:
                print(f"  Modèle: {model.upper()}")

            result = run_single_experiment(
                X_train, X_val, y_train, y_val,
                vectorizer_type='tfidf',
                strategy='split',
                text_columns=text_columns,
                feature_columns=feature_columns,
                max_features=max_features,
                ngram_range=ngram_range,
                model_name=model,
                verbose=False
            )

            all_results.append({
                'strategy': 'Indicateurs + TF-IDF',
                'vectorizer': 'tfidf',
                'model': model,
                'use_features': True,
                'f1_score': result['f1_score'],
                'accuracy': result['accuracy'],
                'train_time': result['train_time'],
                'predict_time': result['predict_time']
            })

            if verbose:
                print(f"    F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}\n")

    # Créer DataFrame et trier par F1
    df_results = pd.DataFrame(all_results).sort_values('f1_score', ascending=False)

    if verbose:
        print(f"\n{'='*80}")
        print("RÉSUMÉ DES RÉSULTATS (triés par F1-Score)")
        print(f"{'='*80}")
        print(df_results.to_string(index=False))
        print(f"{'='*80}\n")

        best = df_results.iloc[0]
        print(f"🏆 MEILLEURE STRATÉGIE: {best['strategy']}")
        print(f"   Modèle: {best['model'].upper()}")
        print(f"   F1-Score: {best['f1_score']:.4f}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"{'='*80}\n")

    return df_results


# Analyse et Visualisation

def analyze_results(
    df_results: pd.DataFrame,
    baseline_f1: Optional[float] = None,
    show_plot: bool = True
) -> None:

    print("\n" + "="*80)
    print("ANALYSE DES RÉSULTATS")
    print("="*80)

    # Statistiques générales
    print(f"\nNombre d'expériences: {len(df_results)}")
    print(f"F1-Score moyen: {df_results['f1_score'].mean():.4f}")
    print(f"F1-Score médian: {df_results['f1_score'].median():.4f}")
    print(f"F1-Score min: {df_results['f1_score'].min():.4f}")
    print(f"F1-Score max: {df_results['f1_score'].max():.4f}")
    print(f"Écart-type: {df_results['f1_score'].std():.4f}")

    # Comparaison avec baseline
    if baseline_f1:
        print(f"\nBaseline F1: {baseline_f1:.4f}")
        better = df_results[df_results['f1_score'] > baseline_f1]
        print(f"Expériences supérieures à baseline: {len(better)} / {len(df_results)}")

        if len(better) > 0:
            best_improvement = (df_results['f1_score'].max() - baseline_f1) / baseline_f1 * 100
            print(f"Meilleure amélioration: +{best_improvement:.2f}%")

    # Analyse par stratégie (si disponible)
    if 'strategy' in df_results.columns:
        print("\n" + "-"*80)
        print("PERFORMANCE PAR STRATÉGIE:")
        print("-"*80)
        strategy_stats = df_results.groupby('strategy')['f1_score'].agg(['mean', 'max', 'count'])
        strategy_stats = strategy_stats.sort_values('mean', ascending=False)
        print(strategy_stats.to_string())

    # Analyse par modèle (si disponible)
    if 'model' in df_results.columns:
        print("\n" + "-"*80)
        print("PERFORMANCE PAR MODÈLE:")
        print("-"*80)
        model_stats = df_results.groupby('model')['f1_score'].agg(['mean', 'max', 'count'])
        model_stats = model_stats.sort_values('mean', ascending=False)
        print(model_stats.to_string())

    # Visualisation
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Graphique 1: Distribution des F1-scores
            ax1 = axes[0]
            ax1.hist(df_results['f1_score'], bins=20, edgecolor='black', alpha=0.7)
            ax1.axvline(df_results['f1_score'].mean(), color='red', linestyle='--',
                       label=f'Moyenne: {df_results["f1_score"].mean():.4f}')
            if baseline_f1:
                ax1.axvline(baseline_f1, color='green', linestyle='--',
                           label=f'Baseline: {baseline_f1:.4f}')
            ax1.set_xlabel('F1-Score')
            ax1.set_ylabel('Fréquence')
            ax1.set_title('Distribution des F1-Scores')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Graphique 2: Comparaison par stratégie ou vectorizer
            ax2 = axes[1]
            if 'strategy' in df_results.columns:
                data = df_results.groupby('strategy')['f1_score'].mean().sort_values()
                data.plot(kind='barh', ax=ax2, color='steelblue')
                ax2.set_xlabel('F1-Score moyen')
                ax2.set_title('Performance par Stratégie')
            elif 'vectorizer' in df_results.columns:
                data = df_results.groupby('vectorizer')['f1_score'].mean().sort_values()
                data.plot(kind='barh', ax=ax2, color='steelblue')
                ax2.set_xlabel('F1-Score moyen')
                ax2.set_title('Performance par Vectorizer')

            ax2.grid(alpha=0.3, axis='x')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\n⚠️  matplotlib non disponible, visualisation skip")

    print("="*80 + "\n")


def run_title_weighting_experiment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    text_columns: List[str] = ['title_clean', 'desc_clean'],
    feature_columns: Optional[List[str]] = None,
    title_weights: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
    vectorizer_type: Literal['count', 'tfidf'] = 'tfidf',
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    model_name: str = 'logreg',
    verbose: bool = True
) -> pd.DataFrame:

    if verbose:
        print(f"\n{'='*80}")
        print("TEST DE PONDÉRATION DU TITRE")
        print(f"{'='*80}")
        print(f"Poids à tester: {title_weights}")
        print(f"Vectorizer: {vectorizer_type.upper()}")
        print(f"Max features: {max_features} (par colonne)")
        print(f"N-gram range: {ngram_range}")
        print(f"Modèle: {model_name.upper()}")
        print(f"{'='*80}\n")

    results = []

    for i, weight in enumerate(title_weights, 1):
        if verbose:
            print(f"\n[{i}/{len(title_weights)}] Test avec title_weight = {weight:.1f}x")
            print("-" * 80)

        # Construire le pipeline avec pondération
        vectorizer = build_split_vectorizer_pipeline(
            vectorizer_type=vectorizer_type,
            text_columns=text_columns,
            feature_columns=feature_columns,
            max_features_title=max_features,
            max_features_desc=max_features,
            ngram_range=ngram_range,
            title_weight=weight
        )

        # Entraîner et évaluer
        pipeline = build_full_pipeline(vectorizer, model_name=model_name)
        result = train_and_evaluate(
            pipeline, X_train, y_train, X_val, y_val, verbose=False
        )

        results.append({
            'title_weight': weight,
            'vectorizer': vectorizer_type,
            'max_features': max_features,
            'ngram_range': str(ngram_range),
            'model': model_name,
            'use_features': feature_columns is not None,
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'train_time': result['train_time'],
            'predict_time': result['predict_time']
        })

        if verbose:
            print(f"  → F1-Score: {result['f1_score']:.4f}")
            print(f"  → Accuracy: {result['accuracy']:.4f}")
            print(f"  → Train time: {result['train_time']:.2f}s")

    df_results = pd.DataFrame(results).sort_values('f1_score', ascending=False)

    if verbose:
        print(f"\n{'='*80}")
        print("RÉSULTATS DE LA PONDÉRATION DU TITRE")
        print(f"{'='*80}")
        print("\n" + df_results[['title_weight', 'f1_score', 'accuracy', 'train_time']].to_string(index=False))

        best = df_results.iloc[0]
        baseline = df_results[df_results['title_weight'] == 1.0]

        if not baseline.empty:
            baseline_f1 = baseline.iloc[0]['f1_score']
            improvement = (best['f1_score'] - baseline_f1) * 100
            print(f"\n{'='*80}")
            print("ANALYSE")
            print(f"{'='*80}")
            print(f"Baseline (weight=1.0): F1 = {baseline_f1:.4f}")
            print(f"Meilleur (weight={best['title_weight']:.1f}): F1 = {best['f1_score']:.4f}")
            print(f"Amélioration: {improvement:+.2f}%")

        print(f"{'='*80}\n")

    return df_results


def save_experiment_results(
    df_results: pd.DataFrame,
    output_path: str = "results/experiment_results.csv",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    import os

    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder les résultats
    df_results.to_csv(output_path, index=False)
    print(f"✓ Résultats sauvegardés: {output_path}")

    # Sauvegarder les métadonnées si fournies
    if metadata:
        import json
        meta_path = output_path.replace('.csv', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Métadonnées sauvegardées: {meta_path}")


def track_all_scores(
    results_dict: Dict[str, pd.DataFrame],
    score_column: str = 'f1_score',
    verbose: bool = True
) -> pd.DataFrame:

    all_results = []

    for exp_name, df in results_dict.items():
        if df is None or df.empty:
            continue

        # Extraire les informations pertinentes
        for idx, row in df.iterrows():
            result = {
                'experiment': exp_name,
                'f1_score': row.get(score_column, 0.0),
                'accuracy': row.get('accuracy', 0.0),
            }

            # Ajouter les paramètres de configuration
            for col in ['vectorizer', 'max_features', 'ngram_range', 'title_weight',
                       'use_features', 'strategy', 'model']:
                if col in row:
                    result[col] = row[col]

            all_results.append(result)

    # Créer DataFrame consolidé
    df_all = pd.DataFrame(all_results)
    df_all = df_all.sort_values('f1_score', ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "="*80)
        print("📊 SUIVI GLOBAL DES SCORES F1")
        print("="*80)
        print(f"\nTotal expériences: {len(df_all)}")
        print(f"\nTop 5 scores:")
        print("-" * 80)

        # Afficher top 5
        for i, row in df_all.head(5).iterrows():
            exp = row['experiment']
            f1 = row['f1_score']
            params = []
            if 'max_features' in row and pd.notna(row['max_features']):
                params.append(f"max_features={int(row['max_features'])}")
            if 'title_weight' in row and pd.notna(row['title_weight']):
                params.append(f"title_weight={row['title_weight']:.1f}")
            if 'use_features' in row and pd.notna(row['use_features']):
                params.append(f"features={'Oui' if row['use_features'] else 'Non'}")

            params_str = ', '.join(params) if params else 'N/A'
            print(f"{i+1}. [{exp}] F1={f1:.4f} | {params_str}")

        print("="*80)

    return df_all


def verify_best_score(
    all_scores_df: pd.DataFrame,
    final_config: Dict[str, Any],
    tolerance: float = 0.0001,
    verbose: bool = True
) -> bool:
    
    if all_scores_df.empty:
        print("⚠️ Aucun score à comparer")
        return False

    best_row = all_scores_df.iloc[0]
    best_score = best_row['f1_score']
    best_exp = best_row['experiment']

    # Essayer de trouver le score de la config finale
    final_score = final_config.get('metadata', {}).get('f1_score', None)
    if final_score is None:
        if verbose:
            print("⚠️ Score F1 non trouvé dans la configuration finale")
        return False

    # Comparer
    is_optimal = abs(final_score - best_score) < tolerance

    if verbose:
        print("\n" + "="*80)
        print("🔍 VÉRIFICATION DE LA CONFIGURATION FINALE")
        print("="*80)
        print(f"\n✓ Meilleur score global: F1 = {best_score:.4f} (de {best_exp})")
        print(f"✓ Score config finale:   F1 = {final_score:.4f}")

        if is_optimal:
            print("\n✅ VALIDATION RÉUSSIE: La configuration finale est optimale!")
        else:
            diff = (best_score - final_score) * 100
            print(f"\n❌ ATTENTION: Score sous-optimal!")
            print(f"   Différence: {diff:+.2f}%")
            print(f"   Meilleur score trouvé dans: {best_exp}")
            print("\n💡 Suggestion: Vérifier si le meilleur score a été correctement intégré")

        print("="*80)

    return is_optimal


def generate_vectorization_report(
    df_vec: pd.DataFrame,
    results_weighting: pd.DataFrame,
    results_strategies: pd.DataFrame,
    grid_results: pd.DataFrame,
    result_tfidf: Dict[str, Any],
    result_merged: Dict[str, Any],
    baseline_f1: float,
    exported_config_path: str = 'results/configs/best_vectorization_config.json',
    save_report: bool = True,
    report_path: str = 'results/vectorization_report.txt',
    verbose: bool = True
) -> Dict[str, Any]:
    
    import json
    from datetime import datetime

    # TRACKING GLOBAL
    all_scores = track_all_scores({
        'exp1_count_vs_tfidf': df_vec,
        'exp2b_title_weighting': results_weighting,
        'exp3_manual_features': results_strategies,
        'exp4_grid_search': grid_results
    }, verbose=False)

    # Sauvegarder tracking
    import os
    os.makedirs('results', exist_ok=True)
    all_scores.to_csv('results/all_scores_tracking.csv', index=False)

    # VÉRIFICATION OPTIMALITÉ
    with open(exported_config_path) as f:
        exported_config = json.load(f)

    is_optimal = verify_best_score(all_scores, exported_config, verbose=False)

    # EXTRACTION DES DONNÉES
    best_overall = all_scores.iloc[0]
    best_from_exp1 = df_vec.iloc[0]
    best_from_exp2b = results_weighting.iloc[0]
    best_from_exp3 = results_strategies.iloc[0]
    best_from_exp4 = grid_results.iloc[0]

    # Calculs
    tfidf_score = best_from_exp1['f1_score']
    count_score = df_vec[df_vec['method'] == 'Count Split']['f1_score'].values[0]
    improvement_1 = ((tfidf_score - count_score) / count_score) * 100

    split_score = result_tfidf['f1_score']
    merged_score = result_merged['f1_score']
    improvement_2 = ((split_score - merged_score) / merged_score) * 100

    baseline_weight = results_weighting[results_weighting['title_weight'] == 1.0]['f1_score'].values[0]
    optimal_weight = best_from_exp2b['title_weight']
    optimal_weight_score = best_from_exp2b['f1_score']
    improvement_3 = ((optimal_weight_score - baseline_weight) / baseline_weight) * 100

    without_features = results_strategies[results_strategies['use_features'] == False]['f1_score'].max()
    with_features = best_from_exp3['f1_score']
    improvement_4 = ((with_features - without_features) / without_features) * 100

    total_improvement = ((best_overall['f1_score'] - baseline_f1) / baseline_f1) * 100

    # GÉNÉRATION DU RAPPORT
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 RAPPORT DE VECTORISATION - PHASE 2")
    report_lines.append("=" * 80)
    report_lines.append(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total expériences: {len(all_scores)}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RÉSUMÉ DES DÉCOUVERTES")
    report_lines.append("=" * 80)

    report_lines.append("\n1️⃣  Count vs TF-IDF")
    report_lines.append(f"   → TF-IDF (F1={tfidf_score:.4f}) surpasse Count (F1={count_score:.4f})")
    report_lines.append(f"   → Amélioration: +{improvement_1:.2f}%")

    report_lines.append("\n2️⃣  Split vs Merged")
    report_lines.append(f"   → Split (F1={split_score:.4f}) surpasse Merged (F1={merged_score:.4f})")
    report_lines.append(f"   → Amélioration: +{improvement_2:.2f}%")

    report_lines.append("\n3️⃣  Pondération du Titre")
    report_lines.append(f"   → Poids optimal: {optimal_weight:.1f}x")
    report_lines.append(f"   → F1 = {optimal_weight_score:.4f} (baseline 1.0x: {baseline_weight:.4f})")
    report_lines.append(f"   → Amélioration: +{improvement_3:.2f}%")

    report_lines.append("\n4️⃣  Features Manuelles")
    report_lines.append(f"   → Sans features: F1 = {without_features:.4f}")
    report_lines.append(f"   → Avec features: F1 = {with_features:.4f}")
    report_lines.append(f"   → Amélioration: +{improvement_4:.2f}%")

    report_lines.append("\n5️⃣  Hyperparamètres Optimaux")
    report_lines.append(f"   → Vectorizer: {best_from_exp4['vectorizer'].upper()}")
    report_lines.append(f"   → Max features: {int(best_from_exp4['max_features']):,}")
    report_lines.append(f"   → N-gram range: {best_from_exp4['ngram_range']}")
    report_lines.append(f"   → Title weight: {best_from_exp4.get('title_weight', 1.0):.1f}x")
    report_lines.append(f"   → Features manuelles: {'Oui' if best_from_exp4['use_features'] else 'Non'}")
    report_lines.append(f"   → F1 Final: {best_from_exp4['f1_score']:.4f}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("🏆 PERFORMANCE FINALE")
    report_lines.append("=" * 80)
    report_lines.append(f"Baseline (raw data, NB01):              F1 = {baseline_f1:.4f}")
    report_lines.append(f"Meilleur score (configuration finale):  F1 = {best_overall['f1_score']:.4f}")
    report_lines.append(f"Amélioration totale:                    +{total_improvement:.2f}%")
    report_lines.append("")
    report_lines.append(f"Statut: {'✅ Configuration optimale validée' if is_optimal else '⚠️  À vérifier'}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("🔍 VÉRIFICATION")
    report_lines.append("=" * 80)
    report_lines.append(f"Config exportée: {exported_config_path}")
    report_lines.append(f"Tracking sauvegardé: results/all_scores_tracking.csv")
    report_lines.append(f"Score config finale: F1 = {exported_config['metadata']['f1_score']:.4f}")
    report_lines.append(f"Meilleur score global: F1 = {best_overall['f1_score']:.4f}")

    if not is_optimal:
        diff = (best_overall['f1_score'] - exported_config['metadata']['f1_score']) * 100
        report_lines.append(f"\n⚠️  ATTENTION: Différence de {diff:.2f}%")
        report_lines.append(f"   Meilleur score trouvé dans: {best_overall['experiment']}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("📋 PROCHAINES ÉTAPES")
    report_lines.append("=" * 80)
    report_lines.append("→ Notebook 03: Sélection du meilleur modèle")
    report_lines.append("→ Phase 3: Intégration des features images")
    report_lines.append("→ Phase 4: Fusion multimodale")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    if save_report:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        if verbose:
            print(f"\n✓ Rapport sauvegardé: {report_path}")

    if verbose:
        print("\n" + report_text)

    return {
        'is_optimal': is_optimal,
        'best_score': best_overall['f1_score'],
        'baseline_score': baseline_f1,
        'total_improvement_pct': total_improvement,
        'all_scores': all_scores,
        'report_text': report_text,
        'findings': {
            'count_vs_tfidf': improvement_1,
            'split_vs_merged': improvement_2,
            'title_weighting': improvement_3,
            'manual_features': improvement_4
        }
    }
