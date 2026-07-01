import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.mini_sam import MiniSAM


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MiniSAM(
        image_size=256,
        patch_size=16,
        embed_dim=256,
        image_encoder_depth=6,
        image_encoder_heads=8,
        mask_decoder_depth=2,
        mask_decoder_heads=8,
        num_multimask_outputs=3,
    ).to(device)

    model.eval()

    b = 2

    images = torch.randn(b, 3, 256, 256, device=device)

    points = torch.tensor(
        [
            [[100.0, 120.0]],
            [[50.0, 60.0]],
        ],
        device=device,
    )

    point_labels = torch.tensor(
        [
            [1],
            [1],
        ],
        device=device,
    )

    boxes = torch.tensor(
        [
            [40.0, 50.0, 180.0, 200.0],
            [30.0, 30.0, 220.0, 230.0],
        ],
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            images=images,
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            masks=None,
            multimask_output=True,
        )

    pred_masks = outputs["masks"]
    pred_ious = outputs["iou_predictions"]

    print("Images             :", images.shape)
    print("Image embeddings   :", outputs["image_embeddings"].shape)
    print("Sparse embeddings  :", outputs["sparse_embeddings"].shape)
    print("Dense embeddings   :", outputs["dense_embeddings"].shape)
    print("Pred masks         :", pred_masks.shape)
    print("Pred IoUs          :", pred_ious.shape)

    assert pred_masks.shape == (b, 3, 256, 256)
    assert pred_ious.shape == (b, 3)

    print("MiniSAM full random forward test passed.")


if __name__ == "__main__":
    main()
