import abc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.distribution import log_normal_diag, log_standard_normal


class Prior(nn.Module, abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        pass


class StandardPrior(Prior):
    """From https://github.com/jmtomczak/intro_dgm"""

    def __init__(self, latent_dim: int, *args: Any, **kwargs: Any):
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return log_standard_normal(
            z, reduction="avg", dim=0
        )  # average over MC samples


class MoGPrior(Prior):
    """From https://github.com/jmtomczak/intro_dgm"""

    def __init__(
        self, latent_dim: int, num_components: int, *args: Any, **kwargs: Any
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

        # params
        self.means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.log_vars = nn.Parameter(torch.randn(num_components, latent_dim))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def get_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.means, self.log_vars

    def sample(self, batch_size: int) -> torch.Tensor:
        means, log_vars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and log_vars
        eps = torch.randn(batch_size, self.latent_dim)
        z = []
        for i in range(batch_size):
            indx = indexes[i]
            z.append(means[[indx]] + eps[[i]] * torch.exp(log_vars[[indx]]))
        return torch.cat(z, dim=0)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        means, log_vars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        mc_samples = z.shape[0]
        z = z.unsqueeze(0)
        means = means.unsqueeze(1)
        log_vars = log_vars.unsqueeze(1)
        log_prob = torch.stack(
            [
                log_normal_diag(z[:, i], means, log_vars)
                for i in range(mc_samples)
            ]
        ).mean(
            dim=0
        )  # average over MC samples
        log_prob = log_prob + torch.log(w)
        return torch.logsumexp(
            log_prob, dim=0, keepdim=False
        )  # batch_size x latent_dim
