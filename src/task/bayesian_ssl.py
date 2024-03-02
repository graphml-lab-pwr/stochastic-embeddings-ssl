import abc
import math
from abc import ABC
from typing import Any, Literal

import torch
import wandb
from torch import nn as nn
from torch.distributions import Distribution

from src.model.prior import Prior
from src.task.ssl import BaseSelfSupervisedLearning
from src.utils.distribution import log_normal_diag


class BaseBayesianSelfSupervisedLearning(BaseSelfSupervisedLearning, ABC):
    MAX_SIGMA = 5
    MIN_SIGMA = 0.01
    MAX_MU = 20
    MIN_MU = -20

    def __init__(
        self,
        encoder: Literal["resnet18", "resnet34", "resnet50"],
        dataset: Literal[
            "mnist", "cifar10", "cifar100", "imagenet", "tinyimagenet"
        ],
        cl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Literal[
            "lambda", "cosine", "cosine_warm_restarts", "one_cycle"
        ],
        prior: Prior,
        sigma: Literal["full_diagonal", "single", "ones"],
        batch_size: int,
        num_classes: int,
        train_iters_per_epoch: int,
        learning_rate: float,
        latent_dim: int,
        hidden_dim: int,
        z_dim: int,
        beta_scale: float,
        mc_samples: int,
        compile_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            encoder,
            dataset,
            cl_loss,
            optimizer,
            scheduler,
            batch_size,
            num_classes,
            train_iters_per_epoch,
            hidden_dim,
            z_dim,
            learning_rate,
            prior=prior,
            sigma=sigma,
            latent_dim=latent_dim,
            beta_scale=beta_scale,
            mc_samples=mc_samples,
            compile_encoder=compile_encoder,
            **kwargs,
        )
        self.init_prior_distribution()
        self.init_fc_rv_layers()

    @property
    def min_log_var(self):
        return 2 * math.log(self.MIN_SIGMA)

    @property
    def max_log_var(self):
        return 2 * math.log(self.MAX_SIGMA)

    @property
    def min_mu(self):
        return self.MIN_MU

    @property
    def max_mu(self):
        return self.MAX_MU

    @abc.abstractmethod
    def init_prior_distribution(self) -> None:
        pass

    @abc.abstractmethod
    def init_fc_rv_layers(self) -> None:
        pass

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        output = self.shared_step(batch)
        self.log_losses(output, train_phase="train")
        self.log_statistics(output, train_phase="train")
        loss = output["loss"]
        assert isinstance(loss, torch.Tensor)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        output = self.shared_step(batch)
        self.log_losses(output, train_phase="val")
        self.log_statistics(output, train_phase="val")

    @abc.abstractmethod
    def shared_step(self, batch: tuple[torch.Tensor, ...]) -> Any:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def project(self, h: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def calculate_loss(
        self,
        h1: torch.Tensor,
        h1_mu: torch.Tensor | None,
        h1_log_var: torch.Tensor | None,
        h2: torch.Tensor,
        h2_mu: torch.Tensor | None,
        h2_log_var: torch.Tensor | None,
        z1: torch.Tensor,
        z1_mu: torch.Tensor | None,
        z1_log_var: torch.Tensor | None,
        z2: torch.Tensor,
        z2_mu: torch.Tensor | None,
        z2_log_var: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def kl_loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    def encoder_log_prob(
        self,
        h: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        return log_normal_diag(h, mu, log_var, reduction="avg", dim=0)

    def projector_log_prob(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        return log_normal_diag(z, mu, log_var, reduction="avg", dim=0)

    @abc.abstractmethod
    def log_statistics(
        self, output: dict[str, Any], train_phase: Literal["train", "val"]
    ):
        pass

    def log_mu_and_sigma(
        self,
        output: dict[str, Any],
        train_phase: Literal["train", "val"],
        rv_name: Literal["h", "z"] = "h",
    ) -> None:
        mu = torch.cat(
            [output[f"q_{rv_name}1"].loc, output[f"q_{rv_name}2"].loc]
        )
        sigma = torch.cat(
            [output[f"q_{rv_name}1"].scale, output[f"q_{rv_name}2"].scale]
        )
        self.log_dict(
            {
                f"{train_phase}_mu_{rv_name}_mean": mu.mean(),
                f"{train_phase}_mu_{rv_name}_std": mu.std(dim=1).mean(),
                f"{train_phase}_sigma_{rv_name}_mean": sigma.mean(),
                f"{train_phase}_sigma_{rv_name}_std": sigma.std(dim=1).mean(),
            },
            sync_dist=True,
        )
        if self.get_wandb_logger() and self.trainer.num_devices == 1:
            wandb.log(
                {
                    f"{train_phase}_mu_{rv_name}_hist": wandb.Histogram(
                        mu.reshape(-1).tolist()
                    ),
                    f"{train_phase}_sigma_{rv_name}_hist": wandb.Histogram(
                        sigma.reshape(-1).tolist()
                    ),
                }
            )

    def log_dist_divergence(
        self,
        output: dict[str, Any],
        train_phase: Literal["train", "val"],
        rv_name: Literal["h", "z"] = "h",
    ) -> None:
        q1, q2 = output[f"q_{rv_name}1"], output[f"q_{rv_name}2"]
        js_divergence = (
            torch.distributions.kl_divergence(q1, q2).sum(-1).mean()
            + torch.distributions.kl_divergence(q2, q1).sum(-1).mean()
        ) / 2
        self.log(f"{train_phase}_{rv_name}_latent_js_div", js_divergence)


class HBayesianSelfSupervisedLearning(BaseBayesianSelfSupervisedLearning):
    """Stochastic on H."""

    def __init__(
        self,
        encoder: Literal["resnet18", "resnet34", "resnet50"],
        dataset: Literal[
            "mnist", "cifar10", "cifar100", "imagenet", "tinyimagenet"
        ],
        cl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Literal[
            "lambda", "cosine", "cosine_warm_restarts", "one_cycle"
        ],
        prior: Prior,
        sigma: Literal["full_diagonal", "single", "ones"],
        batch_size: int,
        num_classes: int,
        train_iters_per_epoch: int,
        learning_rate: float,
        latent_dim: int,
        hidden_dim: int,
        z_dim: int,
        beta_scale: float,
        mc_samples: int,
        compile_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            encoder,
            dataset,
            cl_loss,
            optimizer,
            scheduler,
            prior,
            sigma,
            batch_size,
            num_classes,
            train_iters_per_epoch,
            learning_rate,
            latent_dim,
            hidden_dim,
            z_dim,
            beta_scale,
            mc_samples,
            compile_encoder,
            **kwargs,
        )

    def init_prior_distribution(self) -> None:
        assert hasattr(self.hparams, "prior")
        assert hasattr(self.hparams, "latent_dim")
        self.prior = self.hparams.prior(latent_dim=self.hparams.latent_dim)

    def init_fc_rv_layers(self) -> None:
        assert hasattr(self.hparams, "latent_dim")
        self.fc_mu = nn.Linear(self.encoder_out_dim, self.hparams.latent_dim)
        self.fc_log_var = nn.Linear(
            self.encoder_out_dim, self.hparams.latent_dim
        )

    def shared_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor | Distribution]:
        (x1, x2, _), _ = batch

        repr1, h1, mu1, log_var1, q_h1 = self.forward(x1)
        repr2, h2, mu2, log_var2, q_h2 = self.forward(x2)

        z1 = self.project(h1)
        z2 = self.project(h2)

        loss = self.calculate_loss(
            h1=h1,
            h1_mu=mu1,
            h1_log_var=log_var1,
            h2=h2,
            h2_mu=mu2,
            h2_log_var=log_var2,
            z1=z1,
            z1_mu=None,
            z1_log_var=None,
            z2=z2,
            z2_mu=None,
            z2_log_var=None,
        )

        out = loss | {"q_h1": q_h1, "q_h2": q_h2}

        return out

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Distribution
    ]:
        assert hasattr(self.hparams, "mc_samples") and hasattr(
            self.hparams, "beta_scale"
        )
        x_encoded = self.encoder(x)
        mu = torch.clamp(
            self.fc_mu(x_encoded), min=self.min_mu, max=self.max_mu
        )
        log_var = torch.clamp(
            self.fc_log_var(x_encoded),
            min=self.min_log_var,
            max=self.max_log_var,
        )
        q = torch.distributions.Normal(mu, torch.exp(0.5 * log_var))
        z = q.rsample([self.hparams.mc_samples])
        repr_ = z.mean(dim=0)
        return repr_, z, mu, log_var, q

    def project(
        self, h: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return torch.stack(list(map(self.projection_head, h)))

    def calculate_loss(
        self,
        h1: torch.Tensor,
        h1_mu: torch.Tensor | None,
        h1_log_var: torch.Tensor | None,
        h2: torch.Tensor,
        h2_mu: torch.Tensor | None,
        h2_log_var: torch.Tensor | None,
        z1: torch.Tensor,
        z1_mu: torch.Tensor | None,
        z1_log_var: torch.Tensor | None,
        z2: torch.Tensor,
        z2_mu: torch.Tensor | None,
        z2_log_var: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        assert hasattr(self.hparams, "beta_scale")
        assert isinstance(h1_mu, torch.Tensor) and isinstance(
            h1_log_var, torch.Tensor
        )
        assert isinstance(h2_mu, torch.Tensor) and isinstance(
            h2_log_var, torch.Tensor
        )
        cl_losses = self.cl_loss(z1, z2)
        kl_loss = (
            self.kl_loss(h1, h1_mu, h1_log_var)
            + self.kl_loss(h2, h2_mu, h2_log_var)
        ) / 2
        kl_losses = {
            "kl_prior_loss": kl_loss,
            "weighted_kl_prior_loss": self.hparams.beta_scale * kl_loss,
        }
        final_loss = {
            "loss": cl_losses["cl_loss"] + self.hparams.beta_scale * kl_loss
        }
        return final_loss | cl_losses | kl_losses

    def kl_loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return (
            -(
                self.prior.log_prob(x)
                - self.encoder_log_prob(h=x, mu=mu, log_var=log_var)
            )
            .sum(-1)
            .mean()
        )

    def log_statistics(
        self, output: dict[str, Any], train_phase: Literal["train", "val"]
    ):
        self.log_mu_and_sigma(output, train_phase, rv_name="h")
        if self.prior == "standard":
            self.log_dist_divergence(output, train_phase)


class ZBayesianSelfSupervisedLearning(BaseBayesianSelfSupervisedLearning):
    """Stochastic on Z."""

    def __init__(
        self,
        encoder: Literal["resnet18", "resnet34", "resnet50"],
        dataset: Literal[
            "mnist", "cifar10", "cifar100", "imagenet", "tinyimagenet"
        ],
        cl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Literal[
            "lambda", "cosine", "cosine_warm_restarts", "one_cycle"
        ],
        prior: Prior,
        sigma: Literal["full_diagonal", "single", "ones"],
        batch_size: int,
        num_classes: int,
        train_iters_per_epoch: int,
        learning_rate: float,
        latent_dim: int,
        hidden_dim: int,
        z_dim: int,
        beta_scale: float,
        mc_samples: int,
        compile_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            encoder,
            dataset,
            cl_loss,
            optimizer,
            scheduler,
            prior,
            sigma,
            batch_size,
            num_classes,
            train_iters_per_epoch,
            learning_rate,
            latent_dim,
            hidden_dim,
            z_dim,
            beta_scale,
            mc_samples,
            compile_encoder,
            **kwargs,
        )

    def init_prior_distribution(self) -> None:
        assert hasattr(self.hparams, "prior")
        assert hasattr(self.hparams, "z_dim")
        self.prior = self.hparams.prior(latent_dim=self.hparams.z_dim)

    def init_fc_rv_layers(self) -> None:
        assert hasattr(self.hparams, "z_dim")
        self.fc_mu = nn.Linear(self.hparams.z_dim, self.hparams.z_dim)
        self.fc_log_var = nn.Linear(self.hparams.z_dim, self.hparams.z_dim)

    def shared_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor | Distribution]:
        (x1, x2, _), _ = batch

        h1 = self.forward(x1)
        h2 = self.forward(x2)

        z1_repr, z1, mu1, log_var1, q_z1 = self.project(h1)
        z2_repr, z2, mu2, log_var2, q_z2 = self.project(h2)

        loss = self.calculate_loss(
            h1=h1,
            h1_mu=None,
            h1_log_var=None,
            h2=h2,
            h2_mu=None,
            h2_log_var=None,
            z1=z1,
            z1_mu=mu1,
            z1_log_var=log_var1,
            z2=z2,
            z2_mu=mu2,
            z2_log_var=log_var2,
        )

        out = loss | {"q_z1": q_z1, "q_z2": q_z2}

        return out

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self.encoder(x)

    def project(
        self, h: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Distribution
    ]:
        assert hasattr(self.hparams, "mc_samples")
        if len(h.shape) > 2:
            h = h.mean(dim=0)
        h_encoded = self.projection_head(h)
        mu = torch.clamp(
            self.fc_mu(h_encoded), min=self.min_mu, max=self.max_mu
        )
        log_var = torch.clamp(
            self.fc_log_var(h_encoded),
            min=self.min_log_var,
            max=self.max_log_var,
        )
        q_z = torch.distributions.Normal(mu, torch.exp(0.5 * log_var))
        z = q_z.rsample([self.hparams.mc_samples])
        repr_ = z.mean(dim=0)
        return repr_, z, mu, log_var, q_z

    def calculate_loss(
        self,
        h1: torch.Tensor,
        h1_mu: torch.Tensor | None,
        h1_log_var: torch.Tensor | None,
        h2: torch.Tensor,
        h2_mu: torch.Tensor | None,
        h2_log_var: torch.Tensor | None,
        z1: torch.Tensor,
        z1_mu: torch.Tensor | None,
        z1_log_var: torch.Tensor | None,
        z2: torch.Tensor,
        z2_mu: torch.Tensor | None,
        z2_log_var: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        assert hasattr(self.hparams, "beta_scale")
        assert isinstance(z1_mu, torch.Tensor) and isinstance(
            z1_log_var, torch.Tensor
        )
        assert isinstance(z2_mu, torch.Tensor) and isinstance(
            z2_log_var, torch.Tensor
        )
        cl_losses = self.cl_loss(z1, z2)
        kl_loss = (
            self.kl_loss(z1, z1_mu, z1_log_var)
            + self.kl_loss(z2, z2_mu, z2_log_var)
        ) / 2
        kl_losses = {
            "kl_prior_loss": kl_loss,
            "weighted_kl_prior_loss": self.hparams.beta_scale * kl_loss,
        }
        final_loss = {
            "loss": cl_losses["cl_loss"] + kl_losses["weighted_kl_prior_loss"]
        }
        return final_loss | cl_losses | kl_losses

    def kl_loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return (
            -(
                self.prior.log_prob(x)
                - self.projector_log_prob(z=x, mu=mu, log_var=log_var)
            )
            .sum(-1)
            .mean()
        )

    def log_statistics(
        self, output: dict[str, Any], train_phase: Literal["train", "val"]
    ):
        self.log_mu_and_sigma(output, train_phase, rv_name="z")
        if self.prior == "standard":
            self.log_dist_divergence(output, train_phase)
