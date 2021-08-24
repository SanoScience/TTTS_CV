import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.fpn import FPN
from val_dataloader import FetoscopyDatasetVal


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


model_list = ["model-fold-0_transposed_224.pt",
              "model-fold-1_transposed_224.pt",
              "model-fold-2_transposed_224.pt",
              "model-fold-3_transposed_224.pt",
              "model-fold-4_transposed_224.pt",
              "model-fold-5_transposed_224.pt",
              "model-fold-0_transposed_448.pt",
              "model-fold-1_transposed_448.pt"]

dataset = FetoscopyDatasetVal("../data/*/", x_img_size=224, y_img_size=224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = []

for info in model_list:
    m = FPN(num_blocks=[3, 8, 36, 3], num_classes=4, back_bone="resnet152")
    m.load_state_dict(torch.load(f"../trained_models/{info}", map_location="cpu"))
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


model = Model(models)
for image, mask, name in dataset:
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    mask = cv2.imread(name, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (224, 224))
    colormap = get_colormap()
    mask_rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[mask == cnt] = colormap[cnt]

    output = model(image)
    output = output.detach().squeeze().cpu().numpy()
    output = np.moveaxis(output, 0, -1)
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    ax[0].imshow(output[:, :, 0])
    pred_color_mask = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for c in range(len(colormap)):
        pred_color_mask[np.argmax(output, axis=2) == c] = colormap[c]
    ax[1].imshow(pred_color_mask)
    ax[2].imshow(mask_rgb)
    ax[3].imshow(np.moveaxis(image.detach().squeeze().cpu().numpy(), 0, -1))
    plt.show()
