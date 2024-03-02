import torch
from torch import nn as nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.projection_head(x)
