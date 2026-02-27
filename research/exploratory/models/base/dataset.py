from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Dataset abstrait pour les modèles PyTorch.

    Toutes les implémentations doivent définir __getitem__
    et retourner un dictionnaire prêt pour le modèle.
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
