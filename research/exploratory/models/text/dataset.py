import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset PyTorch pour modèles Transformer avec encodage préalable.

    Attributes:
        encodings (dict[str, torch.Tensor]):
            Dictionnaire contenant les tenseurs encodés produits par le tokenizer
            (input_ids, attention_mask, token_type_ids si présents).
        labels (torch.Tensor):
            Tenseur des labels de classification.
        extra_features (torch.Tensor or None):
            Tenseur optionnel de features supplémentaires associées aux exemples.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        X,
        y,
        extra_features=None,
        max_length: int = 256,
    ):
        """
        Initialise le dataset et effectue l'encodage des textes.

        Args:
            tokenizer (PreTrainedTokenizer):
                Tokenizer Hugging Face pré-entraîné utilisé pour encoder les textes.
            X (Iterable[str]):
                Séquences textuelles à encoder (texte brut ou prétraité).
            y (Iterable[int]):
                Labels de classification associés aux textes.
            extra_features (array-like, optional):
                Features numériques supplémentaires à associer à chaque exemple.
                Par défaut None.
            max_length (int, optional):
                Longueur maximale des séquences après troncature/padding.
                Par défaut 256.
        """
        self.encodings = tokenizer(
            list(X),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        self.labels = torch.tensor(y, dtype=torch.long)

        self.extra_features = (
            torch.tensor(extra_features, dtype=torch.float)
            if extra_features is not None
            else None
        )

    def __getitem__(self, idx):
        """
        Retourne un exemple encodé du dataset.

        Args:
            idx (int): Index de l'exemple à récupérer.

        Returns:
            dict[str, torch.Tensor]:
                Dictionnaire contenant :
                - input_ids
                - attention_mask (et token_type_ids si présents)
                - labels
                - extra_features (si fournies)
        """
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]

        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]

        return item

    def __len__(self):
        """
        Retourne la taille du dataset.

        Returns:
            int: Nombre d'exemples dans le dataset.
        """
        return len(self.labels)
