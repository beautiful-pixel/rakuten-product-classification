from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    """Calcule des features de longueur à partir de champs textuels.

    Ce transformateur extrait des caractéristiques simples correspondant à la
    longueur des textes, mesurée soit en nombre de caractères, soit en nombre de
    mots. Il est compatible avec les pipelines et unions de features
    scikit-learn.

    Args:
        length_unit (str, optional):
            Unité utilisée pour mesurer la longueur du texte. Valeurs possibles :
            ``"char"`` (caractères) ou ``"word"`` (mots). Par défaut ``"char"``.
        cols (tuple[str], optional):
            Noms des colonnes textuelles à analyser. Par défaut
            ``("designation", "description")``.
        name_prefix (str, optional):
            Préfixe optionnel ajouté au nom des features générées (ex.
            ``"char_len"`` ou ``"word_len"``). Par défaut ``None``.
    """

    _ALLOWED_UNITS = {"char", "word"}

    def __init__(
        self,
        length_unit="char",
        cols=("designation", "description"),
        name_prefix=None,
    ):
        if length_unit not in self._ALLOWED_UNITS:
            raise ValueError(
                f"length_unit doit être dans {self._ALLOWED_UNITS}, "
                f"reçu '{length_unit}'."
            )
        self.length_unit = length_unit
        self.cols = cols
        self.name_prefix = name_prefix

    def fit(self, X, y=None):
        """Initialise les noms des features.

        Args:
            X (pandas.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            TextLengthTransformer: Instance ajustée.
        """
        prefix = f"{self.name_prefix}_" if self.name_prefix else ""
        self.feature_names_ = [prefix + col for col in self.cols]
        return self

    def transform(self, X):
        """Calcule les longueurs des champs textuels.

        Args:
            X (pandas.DataFrame): DataFrame contenant les textes.

        Returns:
            pandas.DataFrame:
                DataFrame des longueurs calculées, avec le même index que X.
        """
        X = X.copy().fillna("")

        if self.length_unit == "word":
            data = {
                name: X[col].str.split().str.len()
                for name, col in zip(self.feature_names_, self.cols)
            }
        else:  # char
            data = {
                name: X[col].str.len()
                for name, col in zip(self.feature_names_, self.cols)
            }

        return pd.DataFrame(data, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Retourne les noms des features générées."""
        return np.array(self.feature_names_)
