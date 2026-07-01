import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.image_encoder import MiniImageEncoder


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MiniImageEncoder(
        image_size=256,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        out_channels=256,
    ).to(device)

    x = torch.randn(2, 3, 256, 256, device=device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert y.shape == (2, 256, 16, 16)

    print("Image encoder random test passed.")


if __name__ == "__main__":
    main()