import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MiniSAMDataset
from models.mini_sam import MiniSAM
from losses import MiniSAMLoss


def get_device(device_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def save_checkpoint(path, model, optimizer, epoch, best_val_loss, best_val_iou=0.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_iou": best_val_iou,
        },
        path,
    )

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
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

        loss_dict = criterion(
            pred_masks=outputs["masks"],
            pred_ious=outputs["iou_predictions"],
            gt_masks=gt_masks,
        )

        total_loss += loss_dict["loss"].item()
        total_iou += loss_dict["mean_iou"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_iou": total_iou / max(n_batches, 1),
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, use_amp):
    model.train()

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        gt_masks = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)
        points = batch["point"].to(device, non_blocking=True)
        point_labels = batch["point_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.type == "cuda"),
            dtype=torch.float16,
        ):
            outputs = model(
                images=images,
                points=points,
                point_labels=point_labels,
                boxes=boxes,
                masks=None,
                multimask_output=True,
            )

            loss_dict = criterion(
                pred_masks=outputs["masks"],
                pred_ious=outputs["iou_predictions"],
                gt_masks=gt_masks,
            )

            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_iou += loss_dict["mean_iou"].item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            iou=f"{total_iou / n_batches:.4f}",
        )

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_iou": total_iou / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/oxford_pet/processed")
    parser.add_argument("--out_dir", type=str, default="runs/minisam_oxford_pet")

    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_encoder_depth", type=int, default=6)
    parser.add_argument("--mask_decoder_depth", type=int, default=2)

    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = get_device(args.device_id)
    print("Using device:", device)

    train_dataset = MiniSAMDataset(
        root_dir=args.data_root,
        split="train",
        image_size=args.image_size,
        augment=True,
    )

    val_dataset = MiniSAMDataset(
        root_dir=args.data_root,
        split="val",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
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

    criterion = MiniSAMLoss(
        focal_weight=20.0,
        dice_weight=1.0,
        iou_weight=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(not args.no_amp and device.type == "cuda"),
    )

    best_val_loss = float("inf")
    best_val_iou = 0.0

    print("Train samples:", len(train_dataset))
    print("Val samples  :", len(val_dataset))
    print("Batch size   :", args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=not args.no_amp,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.4f} | "
            f"train IoU {train_stats['mean_iou']:.4f} | "
            f"val loss {val_stats['loss']:.4f} | "
            f"val IoU {val_stats['mean_iou']:.4f}"
        )

        last_ckpt = Path(args.out_dir) / "last.pt"
        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            epoch,
            best_val_loss,
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_ckpt = Path(args.out_dir) / "best.pt"
            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                epoch,
                best_val_loss,
            )
            print(f"Saved best checkpoint: {best_ckpt}")
        
        if val_stats["mean_iou"] > best_val_iou:
            best_val_iou = val_stats["mean_iou"]
            best_iou_ckpt = Path(args.out_dir) / "best_iou.pt"
            save_checkpoint(
                best_iou_ckpt,
                model,
                optimizer,
                epoch,
                best_val_loss,
            )
            print(f"Saved best IoU checkpoint: {best_iou_ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    main()
