"""Inference for 2D US Echocardiography EchoNet dataset."""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.unet import UNet
from models.cenet import CE_Net_OCT
from models.fpn import FPN
import glob


class TTTSInference:
    """EchoCardioInference class."""

    def __init__(self, model_path: str = None
                 ) -> None:
        """
        Args:
            model_path:
        """
        self.model_path = model_path
        self.model = FPN(num_blocks=[2,4,23,3], num_classes=4, back_bone="resnet101")
        self.model.load_state_dict(torch.load(self.model_path,
                                              map_location="cpu"))
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.Resize(size=(448, 448)),
            transforms.ToTensor()
        ])

    def get_visual_prediction(self,
                              image_name,
                              mask_name):
        """
        Args:
            image_name:
            mask_name:
        Returns:
        """
        image = Image.open(image_name)
        mask = Image.open(mask_name)

        size = (448, 448)
        img = self.transforms(image).unsqueeze(0)
        mask = self.transforms(mask).unsqueeze(0)
        pred_mask = self.model(Variable(img))
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = pred_mask.squeeze(0)
        data = pred_mask.cpu().data
        full_mask = torch.argmax(data, 0)
        plt.imshow(full_mask[:, :])
        #pred_mask = transforms.ToPILImage()(data).resize(size)
        #data_mask = mask.squeeze(0).cpu().data
        #mask = transforms.ToPILImage()(data_mask)
        #ig = plt.imshow(pred_mask)
        #ii = plt.imshow(mask.squeeze(0).squeeze(0), alpha=0.4)
        plt.show()


if __name__ == "__main__":
    fet_reg = TTTSInference(model_path="../data/model-fold-0fpa.pt")
    fet_reg.get_visual_prediction(image_name="../data/Video001/images/Video001_frame01250.png",
                                  mask_name="../data/Video001/labels/Video001_frame01250.png")