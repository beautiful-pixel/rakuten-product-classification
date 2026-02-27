
# TokenFrequencyTransformer a supprimer et utiliser les countvectorizer

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TokenFrequencyTransformer(BaseEstimator, TransformerMixin):
    """
    Transformateur permettant de compter la fréquence
    d'apparition de tokens ou motifs dans une série de textes.

    Ce transformateur est générique et peut être utilisé pour :
    - le comptage de balises HTML (ex. "<li>"),
    - le comptage de tokens numériques (ex. "[petite_longueur]"),
    - le comptage d'unités ou de mots-clés spécifiques.

    Il retourne un DataFrame dont chaque colonne correspond à un token,
    et chaque ligne à un document texte.
    """

    def __init__(self, tokens, text_cols=None, normalize=False, name_prefix=None):
        """
        Initialise le transformateur de comptage de tokens.

        Args:
            tokens (list[str]): Liste des tokens ou expressions régulières
                à compter dans les textes.
            normalize (bool, optional): Si True, les fréquences sont
                normalisées par la longueur du texte (en nombre de caractères).
                Par défaut False.
            name_prefix (str | None, optional): Préfixe à ajouter aux noms
                des colonnes générées. Utile pour éviter les collisions
                de noms lors de l'utilisation de FeatureUnion ou ColumnTransformer.
                Par défaut None.
        """
        self.tokens = tokens
        self.text_cols = text_cols
        self.normalize = normalize
        self.name_prefix = name_prefix
        self.prefix = f"{name_prefix}_" if name_prefix else ""

    def fit(self, X, y=None):
        """

        Ce transformateur ne nécessite aucun apprentissage.

        Args:
            X (pd.Series): Série de textes d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            TokenFrequencyTransformer: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Applique le comptage des tokens sur les textes d'entrée.

        Args:
            X (pd.Series): Série pandas contenant les textes à analyser.

        Returns:
            pd.DataFrame: DataFrame contenant les fréquences de chaque token.
            Chaque colonne correspond à un token, et chaque ligne à un texte.
        """
        if self.text_cols:
            text_cols = self.text_cols
        else:
            text_cols = X.dtypes[X.dtypes == 'object'].index
        X[text_cols] = X[text_cols].fillna("")
        text = pd.Series(['']*len(X), index=X.index)
        for col in text_cols:
            text += ' ' + X[col]
        counts = pd.DataFrame(
            {self.prefix + t: text.str.count(t) for t in self.tokens}
        )
        if self.normalize:
            counts = counts.div(text.apply(len), axis=0).fillna(0)
        return counts
    

