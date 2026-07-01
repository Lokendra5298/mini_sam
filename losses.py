import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="mean",
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1
    """

    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )

    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()

    if reduction == "sum":
        return loss.sum()

    return loss


def dice_loss(
    inputs,
    targets,
    eps=1e-6,
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1
    """

    inputs = inputs.sigmoid()

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2.0 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)

    loss = 1.0 - (numerator + eps) / (denominator + eps)

    return loss.mean()


def mask_iou_from_logits(
    inputs,
    targets,
    threshold=0.0,
    eps=1e-6,
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1

    threshold=0.0 means sigmoid(logit) > 0.5.
    """

    pred = (inputs > threshold).float()
    targets = (targets > 0.5).float()

    pred = pred.flatten(1)
    targets = targets.flatten(1)

    intersection = (pred * targets).sum(dim=1)
    union = pred.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)

    return iou


class MiniSAMLoss(nn.Module):
    def __init__(
        self,
        focal_weight=20.0,
        dice_weight=1.0,
        iou_weight=1.0,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_masks,
        pred_ious,
        gt_masks,
    ):
        """
        pred_masks: [B, M, H, W]
        pred_ious:  [B, M]
        gt_masks:   [B, 1, H, W]

        We choose the best mask among M predicted masks using Dice loss.
        """

        b, m, h, w = pred_masks.shape

        gt_masks = gt_masks.float()

        total_focal = 0.0
        total_dice = 0.0
        dice_losses = []

        for i in range(m):
            mask_i = pred_masks[:, i:i + 1, :, :]

            focal_i = sigmoid_focal_loss(mask_i, gt_masks)
            dice_i = dice_loss(mask_i, gt_masks)

            total_focal = total_focal + focal_i
            total_dice = total_dice + dice_i

            dice_losses.append(dice_i.detach())

        total_focal = total_focal / m
        total_dice = total_dice / m

        # For IoU prediction, match every predicted IoU to actual IoU of that mask.
        actual_ious = []
        for i in range(m):
            mask_i = pred_masks[:, i:i + 1, :, :]
            actual_iou_i = mask_iou_from_logits(mask_i, gt_masks)
            actual_ious.append(actual_iou_i)

        actual_ious = torch.stack(actual_ious, dim=1).detach()
        iou_loss = F.mse_loss(pred_ious, actual_ious)

        total_loss = (
            self.focal_weight * total_focal
            + self.dice_weight * total_dice
            + self.iou_weight * iou_loss
        )

        with torch.no_grad():
            best_pred_idx = torch.argmax(actual_ious, dim=1)
            mean_iou = actual_ious.max(dim=1).values.mean()

        return {
            "loss": total_loss,
            "focal_loss": total_focal.detach(),
            "dice_loss": total_dice.detach(),
            "iou_loss": iou_loss.detach(),
            "mean_iou": mean_iou.detach(),
            "best_pred_idx": best_pred_idx.detach(),
        }
