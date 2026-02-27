import torch
import torchvision
from ..base.dataset import BaseDataset


# class ImageDataset(BaseDataset):
#     """
#     Dataset image PyTorch.

#     Args:
#         images (np.ndarray | torch.Tensor)
#         labels (list[int] | None)
#     """

#     def __init__(self, images, labels=None):
#         self.images = torch.tensor(images, dtype=torch.float32)
#         self.labels = labels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         item = {"pixel_values": self.images[idx]}

#         if self.labels is not None:
#             item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

#         return item
    
# class ImageDataset(Dataset):

#     def __init__(self, X, y, transform=None):
#         self.X = X
#         self.y = y
#         self.transform = transform

#     def __getitem__(self, idx):
#         x = torchvision.io.read_image(self.X[idx])
#         y = self.y[idx]
#         if self.transform:
#             x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.X)
    
from torch.utils.data import Dataset
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, img_size=(128,128), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.resize(img, self.img_size)
        img = img.astype("float32") / 255.0
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

