import abc
from typing import Any, Literal

import torch
import torch.nn as nn
from lightly.models.modules import heads
from lightning.pytorch.loggers import WandbLogger

from src.model.encoder import get_encoder
from src.task.base import BaseTask


class BaseSelfSupervisedLearning(BaseTask):
    """Reference https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/barlow-twins.html."""

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
        batch_size: int,
        num_classes: int,
        train_iters_per_epoch: int,
        hidden_dim: int,
        z_dim: int,
        learning_rate: float,
        compile_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__(num_classes, learning_rate)
        if loss_kwargs := cl_loss.keywords.pop("loss_kwargs", None):  # type: ignore
            self.save_hyperparameters(loss_kwargs)
        self.save_hyperparameters()
        self.online_finetuner: nn.Module | None = None
        self.cl_loss = cl_loss()
        self.encoder, self.encoder_out_dim = get_encoder(
            backbone=encoder, dataset=dataset, compile=compile_encoder
        )
        self.init_projection_head(hidden_dim=hidden_dim, z_dim=z_dim)
        self.init_downstream_model()

    def init_downstream_model(self) -> None:
        assert isinstance(self.device, torch.device)
        assert hasattr(self.hparams, "num_classes")
        self.online_finetuner = nn.Linear(
            self.encoder_out_dim, self.hparams.num_classes
        ).to(self.device)

    def init_projection_head(self, hidden_dim: int, z_dim: int) -> None:
        assert hasattr(self.hparams, "dataset")
        if self.hparams.dataset == "imagenet":
            self.projection_head = heads.BarlowTwinsProjectionHead(
                self.encoder_out_dim, hidden_dim, z_dim
            )
        else:
            self.projection_head = heads.ProjectionHead(
                [
                    (
                        self.encoder_out_dim,
                        hidden_dim,
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=False),
                    ),
                    (hidden_dim, z_dim, None, None),
                ]
            )

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        output = self.shared_step(batch)
        self.log_losses(output, train_phase="train")
        loss = output["loss"]
        assert isinstance(loss, torch.Tensor)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        output = self.shared_step(batch)
        self.log_losses(output, train_phase="val")

    @abc.abstractmethod
    def shared_step(self, batch: tuple[torch.Tensor, ...]) -> Any:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def project(self, h: torch.Tensor) -> Any:
        pass

    def on_train_epoch_end(self) -> None:
        self.log("train_epoch", self.current_epoch, sync_dist=True)

    def log_losses(
        self, output: dict[str, Any], train_phase: Literal["train", "val"]
    ) -> None:
        losses = {k: v for k, v in output.items() if k.endswith("loss")}
        for loss_name, loss_val in losses.items():
            if loss_name == "val_loss":
                self.log(
                    f"{train_phase}_{loss_name}",
                    loss_val,
                    prog_bar=True,
                    sync_dist=True,
                )
            else:
                self.log(f"{train_phase}_{loss_name}", loss_val, sync_dist=True)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        assert hasattr(self.hparams, "num_classes")
        assert self.hparams.num_classes is not None
        self.online_finetuner = nn.Linear(
            self.encoder_out_dim, self.hparams.num_classes
        )

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None


class SelfSupervisedLearning(BaseSelfSupervisedLearning):
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
        batch_size: int,
        num_classes: int,
        train_iters_per_epoch: int,
        hidden_dim: int,
        z_dim: int,
        learning_rate: float,
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
            compile_encoder,
            **kwargs,
        )

    def shared_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        (x1, x2, _), _ = batch

        h1 = self.project(self.forward(x1))
        h2 = self.project(self.forward(x2))
        loss = self.cl_loss(h1, h2)
        loss["loss"] = loss["cl_loss"]
        return loss

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self.encoder(x)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        # in case of MI estimation we first unsqueeze the H tensor to match with Monte Carlo losses
        if len(h.shape) == 3:
            h = h.squeeze(dim=0)
        return self.projection_head(h)
