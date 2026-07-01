import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim1, dim2))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
