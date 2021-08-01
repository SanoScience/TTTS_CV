"""Script for segmentation of vessels, fetus, tool and background."""
from comet_ml import Experiment
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time
import torchvision

from models.fpn import FPN
from data_loader import FetoscopyDatasetTrain
from val_dataloader import FetoscopyDatasetVal
from utils import mIOU
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
parser.add_argument("--x_size",
                    type=int,
                    default=224,
                    help="X image size")
parser.add_argument("--y_size",
                    type=int,
                    default=224,
                    help="Y image size")
parser.add_argument("--num_workers",
                    type=int,
                    default=0,
                    help="Number of workers for processing the data")
parser.add_argument("--classes",
                    type=int,
                    default=4,
                    help="Number of classes in the dataset")
parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="Number of batch size")
parser.add_argument("--lr",
                    type=float,
                    default=0.0001,
                    help="Number of learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay")
parser.add_argument("--backbone",
                    type=str,
                    default="resnet152",
                    help="Encoder backbone")
parser.add_argument("--parallel",
                    type=bool,
                    default=False,
                    help="Parallel learning on GPU")
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

dataset = FetoscopyDatasetVal(args.data, x_img_size=args.x_size, y_img_size=args.y_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kfold = KFold(n_splits=6, shuffle=False)

criterion = nn.CrossEntropyLoss()

SMOOTH = 1e-6

train_dataset = FetoscopyDatasetTrain(args.data, x_img_size=args.x_size, y_img_size=args.y_size)
val_dataset = FetoscopyDatasetVal(args.data, x_img_size=args.x_size, y_img_size=args.y_size)

print("--------------------")

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f"FOLD {fold}")
    print("-------------------")
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_subsampler)
    test_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             sampler=test_subsampler)

    # Init neural network
    model = FPN(num_blocks=[3, 8, 36, 3], num_classes=args.classes, back_bone=args.backbone)

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

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
                masks_max = torch.argmax(masks, dim=1)
                output_mask = model(images)
                loss = criterion(output_mask, masks_max)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                jac = mIOU(masks_max, output_mask, num_classes=args.classes)
                running_jaccard += jac.item()
                running_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(" ", end="")
                    print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                          f" Loss: {loss.item():.4f}"
                          f" Jaccard: {jac.item():.4f}"
                          f" Time: {time.time() - start_time_epoch:.2f}s")

            print("Training process has finished. Starting testing...")

            val_running_jac = 0.0
            val_running_loss = 0.0
            best_accuracy = 0.0
            model.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                masks = masks.permute(0, 2, 1, 3)
                masks_max = torch.argmax(masks, dim=1)

                output_mask = model(images)
                loss = criterion(output_mask, masks_max)
                jac = mIOU(masks_max, output_mask, num_classes=args.classes)
                val_running_jac += jac.item()
                val_running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            test_loss = val_running_loss / len(test_loader)

            train_jac = running_jaccard / len(train_loader)
            test_jac = val_running_jac / len(test_loader)

            save_path = f"../data/model-fold-{fold}.pt"

            if best_accuracy < test_jac:
                torch.save(model.state_dict(), save_path)
                best_accuracy = test_jac
                print(f"Model saved!")

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
