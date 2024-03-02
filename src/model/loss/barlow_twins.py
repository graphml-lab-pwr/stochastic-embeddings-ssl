from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        batch_size: int,
        lambda_coeff: float = 5e-3,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x: torch.Tensor) -> torch.Tensor:
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (h1 - torch.mean(h1, dim=0)) / torch.std(h1, dim=0)
        z2_norm = (h2 - torch.mean(h2, dim=0)) / torch.std(h2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return {
            "cl_loss": on_diag + self.lambda_coeff * off_diag,
            "on_diag_loss": on_diag,
            "off_diag_loss": off_diag,
            "weighted_off_diag_loss": self.lambda_coeff * off_diag,
        }


class MCBarlowTwinsLoss(BarlowTwinsLoss):
    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        barlow_twins_loss = defaultdict(list)
        for samples in zip(h1, h2):
            for loss_k, loss_v in super().forward(*samples).items():
                barlow_twins_loss[loss_k].append(loss_v)
        return {
            loss_k: torch.stack(loss_v).mean()
            for loss_k, loss_v in barlow_twins_loss.items()
        }
