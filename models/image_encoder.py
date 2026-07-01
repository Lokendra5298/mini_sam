import torch
import torch.nn as nn

from models.common import MLP, LayerNorm2d


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dim, num_layers=2)

    def forward(self, x):
        # x: [B, N, C]

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class MiniImageEncoder(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        out_channels=256,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
            )
            for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm2d(out_channels),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 3, 256, 256]

        x = self.patch_embed(x)
        # [B, C, 16, 16]

        b, c, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)
        # [B, 256, C]

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        # [B, C, 16, 16]

        x = self.neck(x)
        # [B, 256, 16, 16]

        return x
