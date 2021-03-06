import os
import sys
import torch
import cv2
import numpy as np
from models.fpn import FPN
import torchvision.transforms.functional as F
import glob


INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


def get_colormap():
    """
    Returns FetReg colormap
    """
    colormap = np.asarray(
        [
            [0, 0, 0],  # 0 - background
            [255, 0, 0],  # 1 - vessel
            [0, 0, 255],  # 2 - tool
            [0, 255, 0],  # 3 - fetus

        ]
    )
    return colormap


model_list = ["model-fold-0_transposed_448.pt",
              "model-fold-1_transposed_448.pt",
              "model-fold-2_transposed_448.pt",
              "model-fold-3_transposed_448.pt",
              "model-fold-4_transposed_448.pt",
              "model-fold-5_transposed_448.pt"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = []

for info in model_list:
    m = FPN(num_blocks=[3, 8, 36, 3], num_classes=4, back_bone="resnet152")
    m.load_state_dict(torch.load(f"trained_models/{info}", map_location="cpu"))
    m.to(device)
    m.eval()
    models.append(m)


class Model:
    def __init__(self, models):
        """
        Args:
            models:
        """
        self.models = models

    def __call__(self, x):
        preds = []
        x = x.to(device)

        with torch.no_grad():
            for m in self.models:
                pred = m(x)
                preds.append(pred)
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        return preds


if __name__ == "__main__":

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH + " created")
    else:
        print(OUTPUT_PATH + " exists")

    model = Model(models)
    colormap = get_colormap()
    input_file_list = glob.glob(INPUT_PATH + "/*.png")

    for file in input_file_list:
        file_name = file.split("/")[-1]
        img = cv2.imread(file, 0)
        width, height = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        output = model(img)
        output = output.detach().squeeze().cpu().numpy()
        output = np.moveaxis(output, 0, -1)
        pred_mask = np.argmax(output, axis=2).astype("float32")
        pred_mask = cv2.resize(pred_mask, (width, height))
        pred_mask = np.uint8(pred_mask)
        result = cv2.imwrite(f"{OUTPUT_PATH}/{file_name}", pred_mask)
