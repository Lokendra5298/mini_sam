import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+"
        r"train loss\s+([0-9.]+)\s+\|\s+"
        r"train IoU\s+([0-9.]+)\s+\|\s+"
        r"val loss\s+([0-9.]+)\s+\|\s+"
        r"val IoU\s+([0-9.]+)"
    )

    epochs = []
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_ious.append(float(match.group(3)))
                val_losses.append(float(match.group(4)))
                val_ious.append(float(match.group(5)))

    print(f"Parsed {len(epochs)} epochs from {log_file}")

    if len(epochs) == 0:
        raise RuntimeError("No epoch lines found. Check log file path or format.")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MiniSAM Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_ious, label="Train IoU")
    plt.plot(epochs, val_ious, label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("MiniSAM Training and Validation IoU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "iou_curve.png", dpi=200)
    plt.close()

    best_val_iou = max(val_ious)
    best_val_iou_epoch = epochs[val_ious.index(best_val_iou)]

    best_val_loss = min(val_losses)
    best_val_loss_epoch = epochs[val_losses.index(best_val_loss)]

    print("=" * 60)
    print(f"Best val IoU : {best_val_iou:.4f} at epoch {best_val_iou_epoch}")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
    print(f"Saved curves to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
