import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import LayerNorm2d


class RandomFourierPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=128, scale=10.0):
        super().__init__()
        self.register_buffer(
            "gaussian_matrix",
            scale * torch.randn(2, num_pos_feats),
        )

    def forward(self, coords):
        # coords: [B, N, 2], normalized [0, 1]
        coords = 2.0 * coords - 1.0
        coords = coords @ self.gaussian_matrix
        coords = 2.0 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


class MiniPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        image_embedding_size=(16, 16),
        input_image_size=(256, 256),
        mask_in_channels=16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        self.pe_layer = RandomFourierPositionEmbedding(embed_dim // 2)

        # 0 negative point, 1 positive point, 2 box top-left, 3 box bottom-right
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)
        ])

        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4),
            nn.GELU(),

            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels),
            nn.GELU(),

            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1),
        )

        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def normalize_coords(self, coords):
        h, w = self.input_image_size
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h
        return coords.clamp(0.0, 1.0)

    def embed_points(self, points, labels):
        points = self.normalize_coords(points)
        point_embedding = self.pe_layer(points)

        labels = labels.long()

        point_embedding = torch.where(
            labels[..., None] == -1,
            self.not_a_point_embed.weight[0][None, None, :],
            point_embedding,
        )

        pos_embed = self.point_embeddings[1].weight[0]
        neg_embed = self.point_embeddings[0].weight[0]

        point_embedding = torch.where(
            labels[..., None] == 1,
            point_embedding + pos_embed[None, None, :],
            point_embedding,
        )

        point_embedding = torch.where(
            labels[..., None] == 0,
            point_embedding + neg_embed[None, None, :],
            point_embedding,
        )

        return point_embedding

    def embed_boxes(self, boxes):
        # boxes: [B, 4] = x1, y1, x2, y2
        corners = torch.stack(
            [
                boxes[:, 0:2],
                boxes[:, 2:4],
            ],
            dim=1,
        )

        corners = self.normalize_coords(corners)
        corner_embedding = self.pe_layer(corners)

        corner_embedding[:, 0, :] += self.point_embeddings[2].weight[0]
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight[0]

        return corner_embedding

    def embed_masks(self, masks):
        mask_embedding = self.mask_downscaling(masks)

        if mask_embedding.shape[-2:] != self.image_embedding_size:
            mask_embedding = F.interpolate(
                mask_embedding,
                size=self.image_embedding_size,
                mode="bilinear",
                align_corners=False,
            )

        return mask_embedding

    def get_dense_pe(self, batch_size, device):
        h, w = self.image_embedding_size

        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)

        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)
        coords = coords.reshape(1, h * w, 2).repeat(batch_size, 1, 1)

        pe = self.pe_layer(coords)
        pe = pe.transpose(1, 2).reshape(batch_size, self.embed_dim, h, w)

        return pe

    def forward(
        self,
        points=None,
        point_labels=None,
        boxes=None,
        masks=None,
        batch_size=None,
        device=None,
    ):
        sparse_embeddings = []

        if points is not None:
            sparse_embeddings.append(self.embed_points(points, point_labels))

        if boxes is not None:
            sparse_embeddings.append(self.embed_boxes(boxes))

        if len(sparse_embeddings) > 0:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            assert batch_size is not None
            assert device is not None
            sparse_embeddings = torch.empty(
                batch_size,
                0,
                self.embed_dim,
                device=device,
            )

        if masks is not None:
            dense_embeddings = self.embed_masks(masks)
        else:
            assert batch_size is not None
            assert device is not None

            h, w = self.image_embedding_size
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(batch_size, -1, h, w).to(device)

        return sparse_embeddings, dense_embeddings
