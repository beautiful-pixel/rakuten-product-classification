from langdetect import detect, DetectorFactory
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


DetectorFactory.seed = 0

def detect_lang(text: str) -> str:
    """
    Détecte la langue principale d'un texte.

    Args:
        text (str): Texte à analyser.

    Returns:
        str: Code langue ('fr', 'en', 'other').
    """
    try:
        lang = detect(text)
        return lang if lang in ["fr", "en"] else "other"
    except Exception:
        return "other"

class LanguageDetector(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn extrayant des indicateurs de langue du texte.

    La langue est détectée à partir de la concaténation du titre et de la
    description, puis encodée sous forme de variables binaires.
    """

    def __init__(self, designation_col="designation", description_col="description"):
        """
        Initialise le transformateur de détection de langue.

        Args:
            designation_col (str, optional): Nom de la colonne contenant
                le titre du produit. Par défaut 'designation'.
            description_col (str, optional): Nom de la colonne contenant
                la description du produit. Par défaut 'description'.
        """
        self.designation_col = designation_col
        self.description_col = description_col

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            LanguageDetector: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Détecte la langue du texte et génère des indicateurs binaires.

        Args:
            X (pd.DataFrame): DataFrame contenant les colonnes textuelles.

        Returns:
            pd.DataFrame: DataFrame contenant les variables binaires :
                - txt_fr : texte en français
                - txt_en : texte en anglais
                - txt_other : autre langue
        """
        langs = (
            X[self.designation_col].fillna("") + " " +
            X[self.description_col].fillna("")
        ).apply(detect_lang)

        return pd.DataFrame({
            "txt_fr": (langs == "fr").astype(int),
            "txt_en": (langs == "en").astype(int),
            "txt_other": (langs == "other").astype(int),
        })
