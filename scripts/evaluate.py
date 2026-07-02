import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MiniSAMDataset
from models.mini_sam import MiniSAM


def compute_binary_metrics(pred, gt, eps=1e-6):
    """
    pred: [B, 1, H, W] binary 0/1
    gt:   [B, 1, H, W] binary 0/1
    """

    pred = pred.float().flatten(1)
    gt = gt.float().flatten(1)

    tp = (pred * gt).sum(dim=1)
    fp = (pred * (1.0 - gt)).sum(dim=1)
    fn = ((1.0 - pred) * gt).sum(dim=1)
    tn = ((1.0 - pred) * (1.0 - gt)).sum(dim=1)

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return iou, dice, precision, recall, acc


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/oxford_pet/processed")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_encoder_depth", type=int, default=6)
    parser.add_argument("--mask_decoder_depth", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MiniSAMDataset(
        root_dir=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = MiniSAM(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        image_encoder_depth=args.image_encoder_depth,
        image_encoder_heads=8,
        mask_decoder_depth=args.mask_decoder_depth,
        mask_decoder_heads=8,
        num_multimask_outputs=3,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    selected_ious = []
    selected_dices = []
    selected_precisions = []
    selected_recalls = []
    selected_accs = []

    oracle_ious = []
    oracle_dices = []

    for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
        images = batch["image"].to(device, non_blocking=True)
        gt_masks = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)
        points = batch["point"].to(device, non_blocking=True)
        point_labels = batch["point_label"].to(device, non_blocking=True)

        outputs = model(
            images=images,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            masks=None,
            multimask_output=True,
        )

        pred_logits = outputs["masks"]
        pred_ious = outputs["iou_predictions"]

        # Realistic mask selection: choose mask with highest predicted IoU.
        selected_idx = torch.argmax(pred_ious, dim=1)

        selected_logits = pred_logits[
            torch.arange(pred_logits.shape[0], device=device),
            selected_idx,
        ].unsqueeze(1)

        selected_pred = (selected_logits > 0).float()

        iou, dice, precision, recall, acc = compute_binary_metrics(
            selected_pred,
            gt_masks,
        )

        selected_ious.append(iou.cpu())
        selected_dices.append(dice.cpu())
        selected_precisions.append(precision.cpu())
        selected_recalls.append(recall.cpu())
        selected_accs.append(acc.cpu())

        # Oracle upper-bound: best among 3 masks using actual IoU.
        batch_oracle_ious = []
        batch_oracle_dices = []

        for m in range(pred_logits.shape[1]):
            pred_m = (pred_logits[:, m:m + 1] > 0).float()
            iou_m, dice_m, _, _, _ = compute_binary_metrics(pred_m, gt_masks)
            batch_oracle_ious.append(iou_m)
            batch_oracle_dices.append(dice_m)

        batch_oracle_ious = torch.stack(batch_oracle_ious, dim=1)
        batch_oracle_dices = torch.stack(batch_oracle_dices, dim=1)

        oracle_ious.append(batch_oracle_ious.max(dim=1).values.cpu())
        oracle_dices.append(batch_oracle_dices.max(dim=1).values.cpu())

    selected_ious = torch.cat(selected_ious)
    selected_dices = torch.cat(selected_dices)
    selected_precisions = torch.cat(selected_precisions)
    selected_recalls = torch.cat(selected_recalls)
    selected_accs = torch.cat(selected_accs)

    oracle_ious = torch.cat(oracle_ious)
    oracle_dices = torch.cat(oracle_dices)

    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Samples   : {len(dataset)}")
    print("-" * 70)
    print("Realistic selected mask using predicted IoU:")
    print(f"IoU       : {selected_ious.mean().item():.4f}")
    print(f"Dice      : {selected_dices.mean().item():.4f}")
    print(f"Precision : {selected_precisions.mean().item():.4f}")
    print(f"Recall    : {selected_recalls.mean().item():.4f}")
    print(f"Pixel Acc : {selected_accs.mean().item():.4f}")
    print("-" * 70)
    print("Oracle best mask among 3 candidates:")
    print(f"IoU       : {oracle_ious.mean().item():.4f}")
    print(f"Dice      : {oracle_dices.mean().item():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
