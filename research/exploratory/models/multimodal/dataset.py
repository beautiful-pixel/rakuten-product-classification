import torch
from ..base.dataset import BaseDataset


class MultimodalDataset(BaseDataset):
    """
    Dataset multimodal image + texte.
    """

    def __init__(self, text_dataset, image_dataset):
        assert len(text_dataset) == len(image_dataset)
        self.text_dataset = text_dataset
        self.image_dataset = image_dataset

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, idx):
        text_item = self.text_dataset[idx]
        image_item = self.image_dataset[idx]

        return {
            **text_item,
            **image_item
        }
