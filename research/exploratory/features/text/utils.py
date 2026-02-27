from sklearn.base import BaseEstimator, TransformerMixin


class MergeTextTransformer(BaseEstimator, TransformerMixin):
    """
    Concatène deux colonnes textuelles en une seule.

    Ce transformateur est typiquement utilisé pour fusionner un titre
    et une description en un champ texte unique avant vectorisation.

    Args:
        title_col (str, optional): Nom de la colonne contenant le titre.
            Par défaut "designation".
        desc_col (str, optional): Nom de la colonne contenant la description.
            Par défaut "description".
    """

    def __init__(self, title_col="designation", desc_col="description", sep=None):
        self.title_col = title_col
        self.desc_col = desc_col
        self.sep = sep

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            MergeTextTransformer: Instance du transformateur.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Concatène les colonnes texte spécifiées.

        Les valeurs manquantes sont remplacées par des chaînes vides
        avant la concaténation.

        Args:
            X (pd.DataFrame): DataFrame contenant les colonnes texte.

        Returns:
            np.ndarray: Tableau 1D de textes concaténés.
        """
        title = X[self.title_col].fillna("")
        desc = X[self.desc_col].fillna("")
        full_sep = " " + self.sep + " " if self.sep else " "
        return (title + full_sep + desc).values

    def get_feature_names_out(self, input_features=None):
        return ["text"]
    
    def get_extra_tokens(self):
        if self.sep:
            return [self.sep]
        else:
            return []

class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Applique une pondération multiplicative aux features.

    Ce transformateur permet de modifier l'importance relative d'un
    ensemble de features en les multipliant par un facteur constant.
    Il est typiquement utilisé dans des pipelines sklearn pour
    rééquilibrer des blocs de features (ex: titre vs description).

    Compatible avec les pipelines sklearn et sérialisable.
    
    Args:
        weight (float, optional): Facteur multiplicatif appliqué aux
            features. Par défaut 1.0 (aucune modification).
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (array-like): Features d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            FeatureWeighter: Instance du transformateur.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Applique la pondération aux features.

        Args:
            X (array-like): Matrice de features numériques.

        Returns:
            array-like: Features pondérées.
        """
        return X * self.weight
    
    def get_feature_names_out(self, input_features=None):
        return input_features