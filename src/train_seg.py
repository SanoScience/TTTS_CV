"""Script for segmentation of vessels, fetus, tool and background."""
from comet_ml import Experiment

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time

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
                    default=2,
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

experiment = Experiment("uicx0MlnuGNfKsvBqUHZjPFQx")
experiment.log_parameters(args)

dataset = FetoscopyDataset("../data/*/", x_img_size=224, y_img_size=224)

kfold = KFold(n_splits=6, shuffle=False)

cuda = True if torch.cuda.is_available() else False

criterion = nn.CrossEntropyLoss()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)



print("--------------------")

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f"FOLD {fold}")
    print("-------------------")
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              sampler=train_subsampler)
    test_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             sampler=test_subsampler)

    # Init neural network
    model = UNet(3, 64, 4)
    model = model.cuda() if cuda else model

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)

    with experiment.train():
        for epoch in range(args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            model.train()
            running_loss = 0.0
            running_jaccard = 0.0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = Variable(images.cuda() if cuda else images)
                masks = Variable(masks.cuda() if cuda else masks)
                masks = masks.permute(0, 2, 1, 3)
                masks = torch.argmax(masks, dim=1)

                optimizer.zero_grad()
                output_masks = model(images)
                loss = criterion(output_masks, masks)
                loss.backward()
                optimizer.step()

                jac = dice_coeff(output_masks.round(), masks)
                running_jaccard += jac.item()
                running_loss += loss.item()

                if batch_idx % 20 == 0:
                    mask = masks[0, 0, :]
                    out = output_masks[0, 0, :]
                    res = torch.cat((mask, out), 1).cpu().detach()
                    experiment.log_image(res, name=f"Train: {batch_idx}/{epoch}")

                print(" ", end="")
                print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                      f" Loss: {loss.item():.4f}"
                      f" Jaccard: {jac.item():.4f}"
                      f" Time: {time.time() - start_time_epoch:.2f}s")

            print("Training process has finished. Saving training model...")

            print("Starting testing")

            save_path = f"../data/model-fold-{fold}.pt"
            torch.save(model.state_dict(), save_path)

            val_running_jac = 0.0
            val_running_loss = 0.0
            model.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = Variable(images.cuda() if cuda else images)
                masks = Variable(masks.cuda() if cuda else masks)
                masks = masks.permute(0, 2, 1, 3)

                output_masks = model(images)
                loss = criterion(output_masks, masks)
                jac = dice_coeff(output_masks.round(), masks)
                val_running_jac += jac.item()
                val_running_loss += loss.item()

                if batch_idx % 20 == 0:
                    mask = masks[0, 0, :]
                    out = output_masks[0, 0, :]
                    res = torch.cat((mask, out), 1).cpu().detach()
                    experiment.log_image(res, name=f"Val: {batch_idx}/{epoch}")

            train_loss = running_loss / len(train_loader)
            test_loss = val_running_loss / len(test_loader)

            train_jac = running_jaccard / len(train_loader)
            test_jac = val_running_jac / len(test_loader)
            scheduler.step(test_loss)

            experiment.log_current_epoch(epoch)
            experiment.log_metric("train_jac", train_jac)
            experiment.log_metric("test_jac", test_jac)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("test_loss", test_loss)
            print('    ', end='')

            print(f"Loss: {train_loss:.4f}"
                  f" Train Jaccard: {train_jac:.4f}"
                  f" Test Loss: {test_loss:.4f}"
                  f" Test Jaccard: {test_jac:.4f}")

print(f"Training UNet finished!")
