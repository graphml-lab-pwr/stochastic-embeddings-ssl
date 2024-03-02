import abc
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics import MetricCollection

from src.model.encoder import get_encoder
from src.task.base import BaseTask


class BaseClassification(BaseTask):
    def __init__(
        self,
        model: pl.LightningModule | nn.Module,
        finetune_model: Literal["simple_mlp"],
        encoder_out_dim: int,
        num_classes: int,
        learning_rate: float,
        **kwargs: Any,
    ):
        super().__init__(num_classes, learning_rate)
        self.model = model
        self._init_predictor(finetune_model, encoder_out_dim, num_classes)

    def _init_predictor(
        self, finetune_model: str, encoder_output_dim: int, num_classes: int
    ) -> None:
        assert isinstance(self.device, torch.device)
        if finetune_model == "simple_mlp":
            self.predictor: nn.Module = nn.Linear(
                encoder_output_dim, num_classes
            ).to(self.device)
        else:
            raise ValueError("Invalid finetune model given")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.train_metrics(preds, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.val_metrics(preds, labels)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.test_metrics(preds, labels)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x, _ = self.get_x_y_from_batch(batch)
        return self.forward(x)

    def _shared_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.get_x_y_from_batch(batch)
        preds = self.forward(x)
        loss = F.cross_entropy(preds, y)
        return loss, preds

    @abc.abstractmethod
    def get_x_y_from_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_epoch_end(self) -> None:
        self._aggregate_and_log_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._aggregate_and_log_metrics(self.val_metrics, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self._aggregate_and_log_metrics(self.test_metrics)

    def _aggregate_and_log_metrics(
        self, metrics: MetricCollection, prog_bar: bool = False
    ) -> dict[str, float]:
        metric_values = metrics.compute()
        metrics.reset()
        self.log_dict(metric_values, prog_bar=prog_bar, sync_dist=True)
        return metric_values

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None


class Classification(BaseClassification):
    def __init__(
        self,
        encoder: Literal["resnet18", "resnet34", "resnet50"],
        dataset: Literal[
            "mnist", "cifar10", "cifar100", "imagenet", "tinyimagenet"
        ],
        finetune_model: Literal["simple_mlp"],
        optimizer: torch.optim.Optimizer,
        scheduler: Literal["lambda", "cosine", "cosine_warmup", "one_cycle"],
        num_classes: int,
        learning_rate: float,
        compile_encoder: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("encoder_out_dim", None)
        self.save_hyperparameters()
        model, encoder_out_dim = get_encoder(
            backbone=encoder, dataset=dataset, compile=compile_encoder
        )
        super().__init__(
            model=model,
            finetune_model=finetune_model,
            encoder_out_dim=encoder_out_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            encoder=encoder,
            **kwargs,
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def features_ood(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return self.predictor(out)

    def get_x_y_from_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        return x, y
