import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import MiniSAMDataset
from models.mini_sam import MiniSAM


def tensor_to_image(x):
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    x = x.clip(0, 1)
    return x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/oxford_pet/processed")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="visuals/minisam_predictions")

    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_encoder_depth", type=int, default=6)
    parser.add_argument("--mask_decoder_depth", type=int, default=2)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        shuffle=True,
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

    saved = 0

    for batch in loader:
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

        selected_idx = torch.argmax(pred_ious, dim=1)

        selected_logits = pred_logits[
            torch.arange(pred_logits.shape[0], device=device),
            selected_idx,
        ].unsqueeze(1)

        pred_masks = (selected_logits > 0).float()

        for i in range(images.shape[0]):
            if saved >= args.num_samples:
                print(f"Saved {saved} visualizations to {out_dir}")
                return

            img = tensor_to_image(images[i])
            gt = gt_masks[i, 0].detach().cpu().numpy()
            pred = pred_masks[i, 0].detach().cpu().numpy()

            box = boxes[i].detach().cpu().numpy()
            point = points[i, 0].detach().cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(img)
            axes[0].set_title("Input + Prompt")
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[0].add_patch(rect)
            axes[0].scatter([point[0]], [point[1]], c="red", s=40)

            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("Ground Truth")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title(f"Prediction | IoU score {pred_ious[i, selected_idx[i]].item():.3f}")

            axes[3].imshow(img)
            axes[3].imshow(pred, alpha=0.45, cmap="jet")
            axes[3].set_title("Overlay")

            for ax in axes:
                ax.axis("off")

            save_path = out_dir / f"sample_{saved:03d}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
            plt.close(fig)

            saved += 1

    print(f"Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
