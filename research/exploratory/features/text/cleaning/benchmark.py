import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from .cleaning import clean_text


def load_dataset(data_dir="../data"):

    X_train = pd.read_csv(f"{data_dir}/X_train_update.csv", index_col=0)
    Y_train = pd.read_csv(f"{data_dir}/Y_train_CVw08PX.csv", index_col=0)

    df = X_train.join(Y_train, how="inner")

    # Créer text_raw : designation + " " + description
    df["text_raw"] = (
        df["designation"].fillna("").astype(str).str.strip() + " " +
        df["description"].fillna("").astype(str).str.strip()
    ).str.strip()

    return df



def define_experiments():
    experiments = []

    experiments.append({
        "name": "baseline_raw",
        "group": "0_Baseline",
        "config": {}  # Toutes les options False par défaut
    })

    experiments.append({
        "name": "fix_encoding",
        "group": "1_Encodage",
        "config": {"fix_encoding": True}
    })

    experiments.append({
        "name": "unescape_html",
        "group": "1_Encodage",
        "config": {"unescape_html": True}
    })

    experiments.append({
        "name": "normalize_unicode",
        "group": "1_Encodage",
        "config": {"normalize_unicode": True}
    })

    experiments.append({
        "name": "all_encoding_fixes",
        "group": "1_Encodage",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True
        }
    })

    experiments.append({
        "name": "remove_html_tags",
        "group": "2_HTML",
        "config": {"remove_html_tags": True}
    })

    experiments.append({
        "name": "remove_boilerplate",
        "group": "2_HTML",
        "config": {"remove_boilerplate": True}
    })

    experiments.append({
        "name": "lowercase",
        "group": "3_Casse",
        "config": {"lowercase": True}
    })

    experiments.append({
        "name": "merge_dimensions",
        "group": "4_Fusions",
        "config": {"merge_dimensions": True}
    })

    experiments.append({
        "name": "merge_units",
        "group": "4_Fusions",
        "config": {"merge_units": True}
    })

    experiments.append({
        "name": "merge_durations",
        "group": "4_Fusions",
        "config": {"merge_durations": True}
    })

    experiments.append({
        "name": "merge_age_ranges",
        "group": "4_Fusions",
        "config": {"merge_age_ranges": True}
    })

    experiments.append({
        "name": "tag_years",
        "group": "4_Fusions",
        "config": {"tag_years": True}
    })

    # Combo : Toutes les fusions structurelles
    experiments.append({
        "name": "all_merges",
        "group": "4_Fusions",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    experiments.append({
        "name": "remove_punctuation",
        "group": "5_Ponctuation",
        "config": {"remove_punctuation": True}
    })


    experiments.append({
        "name": "remove_stopwords",
        "group": "6_Filtrage",
        "config": {"remove_stopwords": True}
    })

    experiments.append({
        "name": "remove_single_letters",
        "group": "6_Filtrage",
        "config": {"remove_single_letters": True}
    })

    experiments.append({
        "name": "remove_single_digits",
        "group": "6_Filtrage",
        "config": {"remove_single_digits": True}
    })

    experiments.append({
        "name": "remove_pure_punct_tokens",
        "group": "6_Filtrage",
        "config": {"remove_pure_punct_tokens": True}
    })


    # Approche "clean" traditionnelle
    experiments.append({
        "name": "traditional_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True
        }
    })

    # Approche conservatrice (encodage + HTML seulement)
    experiments.append({
        "name": "conservative_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True
        }
    })

    # Fusions seulement (pas de suppression)
    experiments.append({
        "name": "merges_only",
        "group": "7_Combos",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    return experiments



def run_benchmark(
    df,
    experiments=None,
    test_size=0.15,
    random_state=42,
    tfidf_max_features=10000,
    tfidf_ngram_range=(1, 2),
    verbose=True
):
    if experiments is None:
        experiments = define_experiments()

    if verbose:
        print("=" * 80)
        print("CONFIGURATION DU BENCHMARK")
        print("=" * 80)
        print(f"Total expériences      : {len(experiments)}")
        print(f"Taille de test         : {test_size}")
        print(f"État aléatoire         : {random_state}")
        print(f"TF-IDF max features    : {tfidf_max_features:,}")
        print(f"TF-IDF plage n-grammes : {tfidf_ngram_range}")
        print("=" * 80)
        print()

    # Préparer les labels
    y = df["prdtypecode"].values

    # Créer une seule division train/test (partagée entre toutes les expériences)
    if verbose:
        print("Création de la division train/test...")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    if verbose:
        print(f"  Train : {len(train_idx):,} échantillons")
        print(f"  Test  : {len(test_idx):,} échantillons")
        print()

    # Stocker les résultats
    results = []
    baseline_f1 = None

    # Exécuter les expériences
    for i, exp in enumerate(experiments, 1):
        exp_name = exp["name"]
        exp_group = exp["group"]
        exp_config = exp["config"]

        if verbose:
            print(f"[{i}/{len(experiments)}] {exp_name}")
            print(f"  Groupe : {exp_group}")
            print(f"  Config : {exp_config if exp_config else 'Aucune (données brutes)'}")

        # Appliquer le nettoyage à TOUTES les données d'abord
        if verbose:
            print("  Nettoyage du texte...", end=" ")

        df[f"text_clean_{exp_name}"] = df["text_raw"].apply(
            lambda x: clean_text(x, **exp_config)
        )

        if verbose:
            avg_len = df[f"text_clean_{exp_name}"].str.len().mean()
            print(f"✓ (longueur moyenne : {avg_len:.0f} caractères)")

        # Extraire train/test en utilisant les indices partagés
        X_train_text = df[f"text_clean_{exp_name}"].values[train_idx]
        X_test_text = df[f"text_clean_{exp_name}"].values[test_idx]

        # Construire le pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram_range,
                min_df=2,
                max_df=0.95,
                lowercase=False,  # La fonction de nettoyage gère cela
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs",
                multi_class="multinomial"
            ))
        ])

        # Entraîner
        if verbose:
            print("  Entraînement...", end=" ")
        pipeline.fit(X_train_text, y_train)
        if verbose:
            print("✓")

        # Évaluer
        if verbose:
            print("  Évaluation...", end=" ")
        y_pred = pipeline.predict(X_test_text)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            print("✓")

        # Calculer le delta vs baseline
        if exp_name == "baseline_raw":
            baseline_f1 = f1
            delta_f1 = 0.0
            delta_pct = 0.0
        else:
            delta_f1 = f1 - baseline_f1 if baseline_f1 else 0.0
            delta_pct = (delta_f1 / baseline_f1 * 100) if baseline_f1 else 0.0

        if verbose:
            print(f"  → Score F1 : {f1:.6f} | Exactitude : {acc:.4f}", end="")
            if exp_name != "baseline_raw":
                symbol = "🚀" if delta_f1 > 0 else "📉" if delta_f1 < 0 else "➖"
                print(f" | Δ vs baseline : {symbol} {delta_f1:+.6f} ({delta_pct:+.2f}%)")
            else:
                print(" | [BASELINE]")
            print()

        # Stocker le résultat
        results.append({
            "experiment": exp_name,
            "group": exp_group,
            "f1_weighted": f1,
            "accuracy": acc,
            "delta_f1": delta_f1,
            "delta_pct": delta_pct
        })

        # Nettoyer la colonne temporaire pour économiser la mémoire
        df.drop(columns=[f"text_clean_{exp_name}"], inplace=True)

    if verbose:
        print("=" * 80)
        print("✓ BENCHMARK TERMINÉ")
        print("=" * 80)

    # Créer le DataFrame de résultats
    results_df = pd.DataFrame(results)

    return results_df


def analyze_results(results_df, top_n=10):
    print("\n" + "=" * 80)
    print("ANALYSE DES RÉSULTATS DU BENCHMARK")
    print("=" * 80)
    print()

    # Résumé global
    baseline = results_df[results_df["experiment"] == "baseline_raw"].iloc[0]
    print(f"Score F1 Baseline : {baseline['f1_weighted']:.6f}")
    print()

    # Meilleures améliorations
    print(f"🚀 TOP {top_n} AMÉLIORATIONS :")
    print("-" * 80)
    top_improvements = results_df[results_df["experiment"] != "baseline_raw"].nlargest(top_n, "delta_f1")
    for i, row in top_improvements.iterrows():
        print(f"  {row['experiment']:30s} | F1 : {row['f1_weighted']:.6f} | "
              f"Δ : {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Groupe : {row['group']}")
    print()

    # Moins bonnes performances
    print(f"📉 TOP {top_n} DÉGRADATIONS :")
    print("-" * 80)
    bottom_performers = results_df[results_df["experiment"] != "baseline_raw"].nsmallest(top_n, "delta_f1")
    for i, row in bottom_performers.iterrows():
        print(f"  {row['experiment']:30s} | F1 : {row['f1_weighted']:.6f} | "
              f"Δ : {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Groupe : {row['group']}")
    print()

    # Analyse par groupe
    print("📊 RÉSUMÉ PAR GROUPE :")
    print("-" * 80)
    group_stats = results_df.groupby("group").agg({
        "delta_f1": ["mean", "max", "min"],
        "experiment": "count"
    }).round(6)
    print(group_stats)
    print()

    print("=" * 80)


def save_results(results_df, output_path="results/benchmark_results.csv"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"✓ Résultats sauvegardés dans : {output_path}")
