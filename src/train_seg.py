"""Script for segmentation of vessels, fetus, tool and background."""
from comet_ml import Experiment

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.autograd import Variable
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

experiment = Experiment("uicx0MlnuGNfKsvBqUHZjPFQx")
experiment.log_parameters(args)

dataset = FetoscopyDataset("../data/*/", x_img_size=224, y_img_size=224)

kfold = KFold(n_splits=6, shuffle=False)

cuda = True if torch.cuda.is_available() else False

criterion = DiceLoss()


def Jaccard_index(pred, target):
    intersection = abs(torch.sum(pred * target))
    union = abs(torch.sum(pred + target) - intersection)
    iou = intersection / union
    return iou


def Jaccard_index_multiclass(y_pred, y_true, n_class: int):
    iou = 0.0
    for index in range(n_class):
        iou += Jaccard_index(y_true[:, index, :, :], y_pred[:, index, :, :])  # TODO indexing
    return iou / n_class  # taking average


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
    model = UNet(1, 64, 4)
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

                optimizer.zero_grad()
                output_masks = model(images)
                loss = criterion(output_masks, masks)
                loss.backward()
                optimizer.step()

                jac = Jaccard_index_multiclass(output_masks.round(), masks, n_class=4)
                running_jaccard += jac.item()
                running_loss += loss.item()

                if batch_idx % 1 == 0:
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
                jac = Jaccard_index_multiclass(output_masks.round(), masks, n_class=4)
                val_running_jac += jac.item()
                val_running_loss += loss.item()

                if batch_idx % 1 == 0:
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
