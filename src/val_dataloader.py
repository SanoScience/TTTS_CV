"""Data loader for FetReg: Placental Vessel Segmentation and Registration in Fetoscopy
dataset: https://fetreg2021.grand-challenge.org/"""

import glob
import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset


class FetoscopyDatasetVal(Dataset):
    """FetoscopyDataset class."""
    def __init__(self, data_path, x_img_size, y_img_size) -> None:
        """

        Args:
            data_path:
        """
        self.data_path = data_path
        self.images = glob.glob(self.data_path + "/images/*.png", recursive=True)
        self.masks = glob.glob(self.data_path + "/labels/*.png", recursive=True)
        self.n_classes = 4
        self.x_img_size = x_img_size
        self.y_img_size = y_img_size

        self.images.sort()
        self.masks.sort()

    def __getitem__(self, x) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            x:

        Returns:

        """
        image = Image.open(self.images[x])
        mask = Image.open(self.masks[x])
        resize_transform = transforms.Resize(size=(self.x_img_size, self.y_img_size))
        image = resize_transform(image)
        mask = resize_transform(mask)

        n_mask = np.asarray(mask)

        masks = []
        for class_idx in range(self.n_classes):
            masks.append((n_mask == class_idx).astype(int))
        masks = np.array(masks)

        image = F.to_tensor(image)
        mask = F.to_tensor(masks)

        return image, mask

    def __len__(self) -> int:
        return len(self.images)
