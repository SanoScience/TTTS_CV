"""Script for segmentation of vessels, fetus, tool and background."""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.unet import UNet
from data_loader import FetoscopyDataset
from loss_functions import DiceLoss


parser = argparse.ArgumentParser(description="Training Segmentation Network on Fetal Dataset.")
parser.add_argument("--image_dataset",
                    type=str,
                    default="../data/*/images/*.png",
                    help="Path to the train dataset")
parser.add_argument("--label_dataset",
                    type=str,
                    default="../data/*/labels/*.png",
                    help="Path to the validate dataset")
parser.add_argument("--in_channels",
                    type=int,
                    default=1,
                    help="Number of input channels")
parser.add_argument("--out_channels",
                    type=int,
                    default=64,
                    help="Number of output channels")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="Number of epochs")
parser.add_argument("--num_workers",
                    type=int,
                    default=0,
                    help="Number of workers for processing the data")
parser.add_argument("--classes",
                    type=int,
                    default=1,
                    help="Number of classes in the dataset")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="Number of batch size")
parser.add_argument("--lr",
                    type=float,
                    default=0.0001,
                    help="Number of learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay")
parser.add_argument("--GPU",
                    type=bool,
                    default=True,
                    help="Use GPU")
parser.add_argument("--model_name",
                    type=str,
                    default="fet_reg_model",
                    help="Model name")
args = parser.parse_args()



