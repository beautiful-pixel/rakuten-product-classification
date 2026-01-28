from sklearn.base import BaseEstimator, TransformerMixin

class FillNaTextTransformer(BaseEstimator, TransformerMixin):
    """
    Remplace les valeurs manquantes dans des colonnes textuelles.

    Ce transformateur est utile en amont des pipelines NLP afin de
    garantir l'absence de valeurs manquantes avant les étapes de
    tokenisation ou de vectorisation.

    Args:
        fill_value (str, optional): Valeur utilisée pour remplacer
            les valeurs manquantes. Par défaut chaîne vide "".
    """

    def __init__(self, fill_value=""):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            FillNaTextTransformer: Instance du transformateur.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Remplace les valeurs manquantes par la valeur spécifiée.

        Args:
            X (pd.DataFrame): DataFrame contenant des colonnes textuelles.

        Returns:
            pd.DataFrame: DataFrame sans valeurs manquantes.
        """
        return X.fillna(self.fill_value)
    
    def get_feature_names_out(self, input_features=None):
        return input_features