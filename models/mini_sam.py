import torch
import torch.nn as nn

from models.image_encoder import MiniImageEncoder
from models.prompt_encoder import MiniPromptEncoder
from models.mask_decoder import MiniMaskDecoder


class MiniSAM(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        embed_dim=256,
        image_encoder_depth=6,
        image_encoder_heads=8,
        mask_decoder_depth=2,
        mask_decoder_heads=8,
        num_multimask_outputs=3,
    ):
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.image_embedding_size = (
            image_size // patch_size,
            image_size // patch_size,
        )

        self.image_encoder = MiniImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=image_encoder_depth,
            num_heads=image_encoder_heads,
            out_channels=embed_dim,
        )

        self.prompt_encoder = MiniPromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(image_size, image_size),
        )

        self.mask_decoder = MiniMaskDecoder(
            embed_dim=embed_dim,
            num_heads=mask_decoder_heads,
            transformer_depth=mask_decoder_depth,
            num_multimask_outputs=num_multimask_outputs,
        )

    def forward(
        self,
        images,
        points=None,
        point_labels=None,
        boxes=None,
        masks=None,
        multimask_output=True,
    ):
        """
        images:       [B, 3, 256, 256]
        points:       [B, N, 2]
        point_labels: [B, N]
        boxes:        [B, 4]
        masks:        [B, 1, 256, 256], optional previous mask prompt
        """

        b = images.shape[0]
        device = images.device

        image_embeddings = self.image_encoder(images)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            masks=masks,
            batch_size=b,
            device=device,
        )

        image_pe = self.prompt_encoder.get_dense_pe(
            batch_size=b,
            device=device,
        )

        pred_masks, pred_ious = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return {
            "masks": pred_masks,
            "iou_predictions": pred_ious,
            "image_embeddings": image_embeddings,
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }
