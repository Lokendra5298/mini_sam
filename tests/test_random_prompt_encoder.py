import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.prompt_encoder import MiniPromptEncoder


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = MiniPromptEncoder(
        embed_dim=256,
        image_embedding_size=(16, 16),
        input_image_size=(256, 256),
    ).to(device)

    b = 2

    points = torch.tensor([
        [[100.0, 120.0], [80.0, 90.0]],
        [[50.0, 60.0], [200.0, 210.0]],
    ], device=device)

    point_labels = torch.tensor([
        [1, 0],
        [1, 0],
    ], device=device)

    boxes = torch.tensor([
        [40.0, 50.0, 180.0, 200.0],
        [30.0, 30.0, 220.0, 230.0],
    ], device=device)

    masks = torch.randn(b, 1, 256, 256, device=device)

    sparse, dense = encoder(
        points=points,
        point_labels=point_labels,
        boxes=boxes,
        masks=masks,
        batch_size=b,
        device=device,
    )

    image_pe = encoder.get_dense_pe(batch_size=b, device=device)

    print("Sparse prompt embedding:", sparse.shape)
    print("Dense prompt embedding :", dense.shape)
    print("Image PE               :", image_pe.shape)

    assert sparse.shape == (b, 4, 256)
    assert dense.shape == (b, 256, 16, 16)
    assert image_pe.shape == (b, 256, 16, 16)

    print("Prompt encoder random test passed.")


if __name__ == "__main__":
    main()
