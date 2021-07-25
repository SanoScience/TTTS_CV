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


class FetoscopyDataset(Dataset):
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

        if random.random() > 0.5:
            color_jitter_transform = transforms.ColorJitter(
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.8, 1.2],
                hue=[-0.1, 0.1]
            )
            image = color_jitter_transform.forward(image)

        if random.random() > 0.5:
            (angle, translations, scale, shear) = transforms.RandomAffine.get_params(
                degrees=[-90, 90],
                translate=[0.2, 0.2],
                scale_ranges=[1, 2],
                shears=[-10, 10],
                img_size=[self.x_img_size, self.y_img_size]
            )
            image = F.affine(
                img=image,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0
            )
            mask = F.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0
            )

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        if random.random() > 0.5:
            image = F.gaussian_blur(img=image, kernel_size=[5, 5])

        if random.random() > 0.5:
            image = F.center_crop(image, ([224, 224]))
            mask = F.center_crop(mask, ([224, 224]))

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
