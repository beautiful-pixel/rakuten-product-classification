from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from .replacer import (
    replace_numeric_expressions,
    get_numeric_tokens,
    NUMERIC_GROUPS,
)

COL_TOKENS = ["[TITRE]", "[DESC]"]


class NumericTokensTransformer(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn remplaçant les expressions numériques
    par des tokens sémantiques selon une stratégie paramétrable.

    La stratégie peut être :
    - "light" : normalisation numérique conservative
    - "full"  : normalisation numérique extensive
    - "phys" : 
    - list[str] : liste explicite de types de mesures
    """

    def __init__(
        self,
        designation_col="designation",
        description_col="description",
        strategy="light",
        merge=True
    ):
        """
        Args:
            designation_col (str): Colonne titre.
            description_col (str): Colonne description.
            strategy (str | list[str]): Stratégie numérique ("light", "full")
                ou liste explicite de types de mesures.
            merge (bool): Fusion titre + description.
        """
        self.designation_col = designation_col
        self.description_col = description_col
        self.strategy = strategy
        self.merge = merge

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Applique la normalisation numérique selon la stratégie choisie.
        """
        if isinstance(self.strategy, str):
            mode = self.strategy
        else:
            # stratégie custom → on passe par "custom"
            mode = "custom"

        def _transform_col(series):
            return series.fillna("").apply(
                lambda txt: replace_numeric_expressions(
                    txt,
                    mode=mode if mode != "custom" else None,
                    enabled_measures=self.strategy
                    if isinstance(self.strategy, list)
                    else None,
                )
            )

        title = _transform_col(X[self.designation_col])
        desc = _transform_col(X[self.description_col])

        if self.merge:
            return (
                COL_TOKENS[0] + " " + title
                + " " + COL_TOKENS[1] + " " + desc
            )

        return pd.DataFrame(
            {
                self.designation_col: title,
                self.description_col: desc,
            }
        )

    def get_extra_tokens(self):
        """
        Retourne la liste des tokens numériques utilisés par la stratégie.
        """
        if isinstance(self.strategy, str):
            measure_types = NUMERIC_GROUPS[self.strategy]
        else:
            measure_types = self.strategy

        return COL_TOKENS + get_numeric_tokens(measure_types)
