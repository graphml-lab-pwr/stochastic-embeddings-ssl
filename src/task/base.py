import abc
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightly.utils.scheduler import CosineWarmupScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torchmetrics import Accuracy, CalibrationError, MetricCollection

from src.task.utils import linear_warmup_decay


class BaseTask(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.init_metrics(num_classes)

    def init_metrics(self, num_classes: int) -> None:
        self.metrics = self.get_default_metrics(num_classes)
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

    def get_default_metrics(self, num_classes: int) -> MetricCollection:
        # noinspection PyTypeChecker
        task: Literal["binary", "multiclass"] = (
            "binary" if num_classes == 2 else "multiclass"
        )

        acc = Accuracy(num_classes=num_classes, task=task)
        acc_at_5 = Accuracy(num_classes=num_classes, task=task, top_k=5)
        ece = CalibrationError(num_classes=num_classes, task=task, norm="l1")

        return MetricCollection(
            {
                "accuracy": acc,  # type: ignore
                "accuracy_at_5": acc_at_5,  # type: ignore
                "ece": ece,  # type: ignore
            }
        )

    def configure_optimizers(self) -> Any:
        assert hasattr(self.hparams, "scheduler") and hasattr(
            self.hparams, "optimizer"
        )
        self.hparams.optimizer.keywords.pop("name")
        scale_lr = self.hparams.optimizer.keywords.pop("scale_lr")
        if scale_lr:
            assert hasattr(self.hparams, "batch_size")
            lr_factor = self.hparams.batch_size / 256
            self.hparams.optimizer.keywords["lr"] *= lr_factor
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            scheduler = self.get_scheduler(optimizer=optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def get_scheduler(self, optimizer: Optimizer) -> dict[str, Any]:
        assert hasattr(self.hparams, "scheduler")
        if self.hparams.scheduler == "lambda":
            assert hasattr(self.hparams, "train_iters_per_epoch")
            assert hasattr(self.hparams, "warmup_epochs")
            warmup_steps = (
                self.hparams.train_iters_per_epoch * self.hparams.warmup_epochs
            )
            return {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=linear_warmup_decay(warmup_steps),
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.scheduler == "cosine":
            assert isinstance(self.trainer.max_epochs, int)
            assert hasattr(self.hparams, "sch_eta_min_scale")
            T_max = self.trainer.max_epochs
            eta_min = self.learning_rate * self.hparams.sch_eta_min_scale
            return {
                "scheduler": CosineAnnealingLR(
                    optimizer=optimizer, T_max=T_max, eta_min=eta_min
                ),
            }
        elif self.hparams.scheduler == "cosine_warm_restarts":
            assert isinstance(self.trainer.max_epochs, int)
            assert hasattr(self.hparams, "sch_eta_min_scale")
            T_0 = self.trainer.max_epochs // 10
            T_mult = 2
            eta_min = self.learning_rate * self.hparams.sch_eta_min_scale
            return {
                "scheduler": CosineAnnealingWarmRestarts(
                    optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
                ),
            }
        elif self.hparams.scheduler == "cosine_warmup":
            assert hasattr(self.hparams, "sch_eta_min_scale")
            assert self.trainer.estimated_stepping_batches is not None
            assert self.trainer.max_epochs is not None
            return {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=int(
                        self.trainer.estimated_stepping_batches
                        / self.trainer.max_epochs
                        * 4
                    ),
                    end_value=self.hparams.sch_eta_min_scale,
                    max_epochs=int(self.trainer.estimated_stepping_batches),
                ),
                "interval": "step",
                "frequency": 1,
            }

        elif self.hparams.scheduler == "one_cycle":
            assert hasattr(self.trainer, "datamodule")
            assert isinstance(self.trainer.max_epochs, int)
            return {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    max_lr=self.learning_rate,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=len(
                        self.trainer.datamodule.train_dataloader()
                    ),
                ),
            }
        else:
            raise ValueError("Wrong name given for lr_scheduler")
