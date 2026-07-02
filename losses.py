import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss_per_sample(
    inputs,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1

    returns: [B]
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

    loss = loss.flatten(1).mean(dim=1)

    return loss


def dice_loss_per_sample(
    inputs,
    targets,
    eps=1e-6,
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1

    returns: [B]
    """

    inputs = inputs.sigmoid()

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2.0 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)

    loss = 1.0 - (numerator + eps) / (denominator + eps)

    return loss


def mask_iou_from_logits(
    inputs,
    targets,
    threshold=0.0,
    eps=1e-6,
):
    """
    inputs:  [B, 1, H, W] logits
    targets: [B, 1, H, W] binary 0/1

    returns: [B]
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

        SAM-style behavior:
        - compute loss for each predicted mask
        - choose best mask per sample
        - backprop only through selected best mask
        - train IoU head to predict actual IoU for every mask
        """

        b, m, h, w = pred_masks.shape
        gt_masks = gt_masks.float()

        all_mask_losses = []
        all_focal_losses = []
        all_dice_losses = []
        all_actual_ious = []

        for i in range(m):
            mask_i = pred_masks[:, i:i + 1, :, :]

            focal_i = sigmoid_focal_loss_per_sample(mask_i, gt_masks)
            dice_i = dice_loss_per_sample(mask_i, gt_masks)

            mask_loss_i = (
                self.focal_weight * focal_i
                + self.dice_weight * dice_i
            )

            actual_iou_i = mask_iou_from_logits(mask_i, gt_masks)

            all_mask_losses.append(mask_loss_i)
            all_focal_losses.append(focal_i)
            all_dice_losses.append(dice_i)
            all_actual_ious.append(actual_iou_i)

        all_mask_losses = torch.stack(all_mask_losses, dim=1)
        all_focal_losses = torch.stack(all_focal_losses, dim=1)
        all_dice_losses = torch.stack(all_dice_losses, dim=1)
        all_actual_ious = torch.stack(all_actual_ious, dim=1)

        # Choose best mask per sample.
        # Lower mask loss = better mask.
        best_idx = torch.argmin(all_mask_losses.detach(), dim=1)

        batch_indices = torch.arange(b, device=pred_masks.device)

        selected_mask_loss = all_mask_losses[batch_indices, best_idx].mean()
        selected_focal_loss = all_focal_losses[batch_indices, best_idx].mean()
        selected_dice_loss = all_dice_losses[batch_indices, best_idx].mean()
        selected_iou = all_actual_ious[batch_indices, best_idx].mean()

        # Train IoU head for all masks.
        iou_loss = F.mse_loss(pred_ious, all_actual_ious.detach())

        total_loss = selected_mask_loss + self.iou_weight * iou_loss

        return {
            "loss": total_loss,
            "focal_loss": selected_focal_loss.detach(),
            "dice_loss": selected_dice_loss.detach(),
            "iou_loss": iou_loss.detach(),
            "mean_iou": selected_iou.detach(),
            "best_pred_idx": best_idx.detach(),
        }