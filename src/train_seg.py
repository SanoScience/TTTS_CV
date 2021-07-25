"""Script for segmentation of vessels, fetus, tool and background."""
from comet_ml import Experiment
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time

from models.fpn import FPN
from data_loader import FetoscopyDataset
import torch.onnx

parser = argparse.ArgumentParser(description="Training Segmentation Network on Fetal Dataset.")
parser.add_argument("--data",
                    type=str,
                    default="../data/*/",
                    help="Path to the data")
parser.add_argument("--in_channels",
                    type=int,
                    default=3,
                    help="Number of input channels")
parser.add_argument("--out_channels",
                    type=int,
                    default=64,
                    help="Number of output channels")
parser.add_argument("--epochs",
                    type=int,
                    default=100,
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
                    default=4,
                    help="Number of biatch size")
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

dataset = FetoscopyDataset(args.data, x_img_size=448, y_img_size=448)

kfold = KFold(n_splits=6, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

SMOOTH = 1e-6


def mIOU(label, pred, num_classes=4):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


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
    model = FPN(num_blocks=[3, 8, 36, 3], num_classes=4, back_bone="resnet152")
    model = model.to(device)

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
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                masks = masks.permute(0, 2, 1, 3)
                masks = torch.argmax(masks, dim=1)

                output_mask = model(images)
                loss = criterion(output_mask, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                jac = mIOU(masks, output_mask, num_classes=4)
                running_jaccard += jac.item()
                running_loss += loss.item()

                print(" ", end="")
                print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                      f" Loss: {loss.item():.4f}"
                      f" Jaccard: {jac.item():.4f}"
                      f" Time: {time.time() - start_time_epoch:.2f}s")

            print("Training process has finished. Saving training model...")

            print("Starting testing")

            save_path = f"../data/model-fold-{fold}.pt"
            torch.save(model.state_dict(), save_path)
            #torch.onnx.export(model, images, f"../data/model-fold-{fold}.onnx")
            val_running_jac = 0.0
            val_running_loss = 0.0
            model.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                masks = masks.permute(0, 2, 1, 3)
                masks = torch.argmax(masks, dim=1)

                output_mask = model(images)
                loss = criterion(output_mask, masks)
                jac = mIOU(masks, output_mask, num_classes=4)
                val_running_jac += jac.item()
                val_running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            test_loss = val_running_loss / len(test_loader)

            train_jac = running_jaccard / len(train_loader)
            test_jac = val_running_jac / len(test_loader)
            scheduler.step(test_loss)

            experiment.log_current_epoch(epoch)
            experiment.log_metric("train_jac", train_jac)
            experiment.log_metric("val_jac", test_jac)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("val_loss", test_loss)

            print('    ', end='')

            print(f"Loss: {train_loss:.4f}"
                  f" Train Jaccard: {train_jac:.4f}"
                  f" Test Loss: {test_loss:.4f}"
                  f" Test Jaccard: {test_jac:.4f}")

print(f"Training UNet finished!")
