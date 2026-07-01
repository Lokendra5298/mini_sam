import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from losses import MiniSAMLoss


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    b = 2
    m = 3
    h = 256
    w = 256

    pred_masks = torch.randn(b, m, h, w, device=device)
    pred_ious = torch.rand(b, m, device=device)

    gt_masks = torch.randint(
        low=0,
        high=2,
        size=(b, 1, h, w),
        device=device,
    ).float()

    criterion = MiniSAMLoss(
        focal_weight=20.0,
        dice_weight=1.0,
        iou_weight=1.0,
    ).to(device)

    outputs = criterion(
        pred_masks=pred_masks,
        pred_ious=pred_ious,
        gt_masks=gt_masks,
    )

    print("Total loss :", float(outputs["loss"]))
    print("Focal loss :", float(outputs["focal_loss"]))
    print("Dice loss  :", float(outputs["dice_loss"]))
    print("IoU loss   :", float(outputs["iou_loss"]))
    print("Mean IoU   :", float(outputs["mean_iou"]))
    print("Best idx   :", outputs["best_pred_idx"])

    assert torch.isfinite(outputs["loss"])

    print("Random loss test passed.")


if __name__ == "__main__":
    main()
