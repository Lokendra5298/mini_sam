import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.prompt_encoder import MiniPromptEncoder
from models.mask_decoder import MiniMaskDecoder


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    b = 2
    c = 256
    h = 16
    w = 16

    prompt_encoder = MiniPromptEncoder(
        embed_dim=c,
        image_embedding_size=(h, w),
        input_image_size=(256, 256),
    ).to(device)

    mask_decoder = MiniMaskDecoder(
        embed_dim=c,
        num_heads=8,
        transformer_depth=2,
        num_multimask_outputs=3,
    ).to(device)

    image_embeddings = torch.randn(b, c, h, w, device=device)

    points = torch.tensor([
        [[100.0, 120.0]],
        [[50.0, 60.0]],
    ], device=device)

    point_labels = torch.tensor([
        [1],
        [1],
    ], device=device)

    boxes = torch.tensor([
        [40.0, 50.0, 180.0, 200.0],
        [30.0, 30.0, 220.0, 230.0],
    ], device=device)

    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=points,
        point_labels=point_labels,
        boxes=boxes,
        masks=None,
        batch_size=b,
        device=device,
    )

    image_pe = prompt_encoder.get_dense_pe(
        batch_size=b,
        device=device,
    )

    with torch.no_grad():
        masks, iou_predictions = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

    print("Image embeddings:", image_embeddings.shape)
    print("Sparse prompts  :", sparse_embeddings.shape)
    print("Dense prompts   :", dense_embeddings.shape)
    print("Masks           :", masks.shape)
    print("IoU predictions :", iou_predictions.shape)

    assert masks.shape == (b, 3, 256, 256)
    assert iou_predictions.shape == (b, 3)

    print("Mask decoder random test passed.")


if __name__ == "__main__":
    main()
