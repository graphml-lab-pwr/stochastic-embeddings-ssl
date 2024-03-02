from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    variance_loss_epsilon = 1e-6

    def __init__(
        self,
        batch_size: int,
        z_dim: int,
        variance_coeff: float,
        invariance_coeff: float,
        covariance_coeff: float,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_features = z_dim
        self.variance_coeff = variance_coeff
        self.invariance_coeff = invariance_coeff
        self.covariance_coeff = covariance_coeff

    def forward(
        self, z_a: torch.Tensor, z_b: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        loss_var = self.variance_loss(z_a, z_b)
        loss_inv = self.invariance_loss(z_a, z_b)
        loss_cov = self.covariance_loss(z_a, z_b)

        weighted_var = loss_var * self.variance_coeff
        weighted_inv = loss_inv * self.invariance_coeff
        weighted_cov = loss_cov * self.covariance_coeff

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "cl_loss": loss,
            "var_loss": loss_var,
            "weighted_var_loss": weighted_var,
            "inv_loss": loss_inv,
            "weighted_inv_loss": weighted_inv,
            "cov_loss": loss_cov,
            "weighted_cov_loss": weighted_cov,
        }

    def variance_loss(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        var_z_a = torch.sqrt(z_a.var(dim=0) + self.variance_loss_epsilon)
        var_z_b = torch.sqrt(z_b.var(dim=0) + self.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - var_z_a))
        loss_v_b = torch.mean(F.relu(1 - var_z_b))
        return (loss_v_a + loss_v_b) / 2

    def invariance_loss(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(z_a, z_b)

    def covariance_loss(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
        loss_c_a = (
            self.off_diagonal(cov_z_a).pow_(2).sum().div(self.num_features)
        )
        loss_c_b = (
            self.off_diagonal(cov_z_b).pow_(2).sum().div(self.num_features)
        )
        return loss_c_a + loss_c_b

    @staticmethod
    def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MCVICRegLoss(VICRegLoss):
    def forward(
        self, z_a: torch.Tensor, z_b: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        vicreg_loss = defaultdict(list)
        for samples in zip(z_a, z_b):
            for loss_k, loss_v in super().forward(*samples).items():
                vicreg_loss[loss_k].append(loss_v)
        return {
            loss_k: torch.stack(loss_v).mean()
            for loss_k, loss_v in vicreg_loss.items()
        }
