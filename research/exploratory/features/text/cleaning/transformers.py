from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .cleaning import clean_text


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

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn appliquant un nettoyage de base du texte.

    Ce transformateur applique une série d'opérations de normalisation
    (encodage, HTML, Unicode) sur les colonnes textuelles spécifiées.
    """

    def __init__(
            self, 
            cols=['designation', 'description'], 
            name_prefix=None, 
            lowercase=False,
            remove_html_tags=True
            ):
        """
        Initialise le transformateur de nettoyage du texte.

        Args:
            cols (list[str], optional): Liste des colonnes textuelles à nettoyer.
                Par défaut ['designation', 'description'].
            name_prefix (str | None, optional): Préfixe ajouté aux noms
                des colonnes générées. Par défaut None.
        """
        self.cols = cols
        self.name_prefix = name_prefix
        self.prefix = f"{name_prefix}_" if name_prefix else ""
        self.lowercase = lowercase
        self.remove_html_tags = remove_html_tags

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            TextCleaner: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Applique le nettoyage du texte aux colonnes spécifiées.

        Args:
            X (pd.DataFrame): DataFrame contenant les colonnes textuelles.

        Returns:
            pd.DataFrame: DataFrame contenant les textes nettoyés.
        """
        params = {
            'fix_encoding':True, 'unescape_html':True,
            'normalize_unicode':True, 'remove_html_tags':self.remove_html_tags,
            'lowercase':self.lowercase,
            }
        cleaned_text = {
            self.prefix + c : X[c].fillna("").apply(lambda x : clean_text(x, **params))
            for c in self.cols
            }
        return pd.DataFrame(cleaned_text)

    def get_feature_names_out(self, input_features=None):
        return input_features

