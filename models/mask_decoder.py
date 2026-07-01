import torch
import torch.nn as nn

from models.common import MLP, LayerNorm2d


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.token_self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.token_to_image_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim,
            int(embed_dim * mlp_ratio),
            embed_dim,
            num_layers=2,
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.image_to_token_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, tokens, image_tokens):
        q = self.norm1(tokens)
        attn_out, _ = self.token_self_attn(q, q, q, need_weights=False)
        tokens = tokens + attn_out

        q = self.norm2(tokens)
        attn_out, _ = self.token_to_image_attn(
            q,
            image_tokens,
            image_tokens,
            need_weights=False,
        )
        tokens = tokens + attn_out

        tokens = tokens + self.mlp(self.norm3(tokens))

        q = self.norm4(image_tokens)
        attn_out, _ = self.image_to_token_attn(
            q,
            tokens,
            tokens,
            need_weights=False,
        )
        image_tokens = image_tokens + attn_out

        return tokens, image_tokens


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth=2,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.final_attn_token_to_image = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, image_embedding, image_pe, tokens):
        b, c, h, w = image_embedding.shape

        image_tokens = image_embedding.flatten(2).transpose(1, 2)
        image_pe_tokens = image_pe.flatten(2).transpose(1, 2)

        image_tokens = image_tokens + image_pe_tokens

        for layer in self.layers:
            tokens, image_tokens = layer(tokens, image_tokens)

        q = self.norm_final(tokens)
        attn_out, _ = self.final_attn_token_to_image(
            q,
            image_tokens,
            image_tokens,
            need_weights=False,
        )
        tokens = tokens + attn_out

        image_tokens = image_tokens.transpose(1, 2).reshape(b, c, h, w)

        return tokens, image_tokens


class MiniMaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        transformer_depth=2,
        num_multimask_outputs=3,
        hypernet_dim=32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)

        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, hypernet_dim, kernel_size=2, stride=2),
            LayerNorm2d(hypernet_dim),
            nn.GELU(),

            nn.ConvTranspose2d(hypernet_dim, hypernet_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, hypernet_dim, num_layers=3)
            for _ in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(
            embed_dim,
            256,
            self.num_mask_tokens,
            num_layers=3,
        )

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=True,
    ):
        b = image_embeddings.shape[0]

        output_tokens = torch.cat(
            [
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            dim=0,
        )

        output_tokens = output_tokens.unsqueeze(0).expand(b, -1, -1)

        tokens = torch.cat(
            [output_tokens, sparse_prompt_embeddings],
            dim=1,
        )

        src = image_embeddings + dense_prompt_embeddings

        hs, src = self.transformer(
            image_embedding=src,
            image_pe=image_pe,
            tokens=tokens,
        )

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]

        upscaled_embedding = self.output_upscaling(src)

        b, c, h, w = upscaled_embedding.shape

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )

        hyper_in = torch.stack(hyper_in_list, dim=1)

        upscaled_embedding_flat = upscaled_embedding.view(b, c, h * w)
        masks = torch.matmul(hyper_in, upscaled_embedding_flat)
        masks = masks.view(b, self.num_mask_tokens, h, w)

        iou_predictions = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            masks = masks[:, 0:1, :, :]
            iou_predictions = iou_predictions[:, 0:1]

        return masks, iou_predictions
