"""
Models for evaluation of SSL-trained models (probabilistic and non-probabilistic):
    1. Linear evaluation (use LinearProbingClassifier with frozen_encoder=True)
    2. Transfer learning (use LinearProbingClassifier with frozen_encoder=True)
    3. Semi-supervised fine-tuning (use LinearProbingClassifier with frozen_encoder=False)
"""

from abc import ABC
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MultilabelAveragePrecision,
)
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from src.utils.utils import load_model_from_ckpt


class LinearProbingBase(ABC, nn.Module):
    def __init__(self, input_dim: int, num_classes: int, loss, metric):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.loss = loss
        self.metric = metric

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class MulticlassLinear(LinearProbingBase):
    def __init__(self, input_dim: int, num_classes: int, top_k: int = 1):
        loss = nn.CrossEntropyLoss()
        metric = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=num_classes, top_k=top_k
                ),
                "ece": MulticlassCalibrationError(num_classes=num_classes),
            }
        )
        super().__init__(input_dim, num_classes, loss, metric)


class MultilabelLinear(LinearProbingBase):
    def __init__(self, input_dim: int, num_classes: int):
        loss = nn.MultiLabelSoftMarginLoss()
        metric = MetricCollection(
            {
                "mAP": MultilabelAveragePrecision(
                    num_labels=num_classes, average="macro"
                ),
            }
        )
        super().__init__(input_dim, num_classes, loss, metric)

    def forward(self, x: Tensor) -> Tensor:
        logits = super().forward(x)
        return logits.sigmoid()


class LinearProbingClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_path: Path | None,
        optim_kwargs: dict[str, Any],
        max_epochs: int,
        scheduler_kwargs: dict[str, Any] | None,
        num_classes: int,
        task: Literal["multiclass", "multilabel"],
        frozen_encoder: bool = True,
        norm_repr: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.frozen_encoder = frozen_encoder
        self.norm_repr = norm_repr

        self.encoder, self.repr_dim = self._get_encoder(model_name, model_path)
        self.predictor = self._get_predictor(task, self.repr_dim, num_classes)

        self.metrics = nn.ModuleDict(
            {
                f"{split}_metrics": self.predictor.metric.clone(
                    prefix=f"{split}/"
                )
                for split in ["train", "val", "test"]
            }
        )

    def _get_encoder(
        self, model_name: str, model_path: Path | None
    ) -> tuple[nn.Module, int]:
        repr_dim = 512
        if model_name == "resnet-50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Identity()  # replace last layer to get representation
            repr_dim = 2048
        elif model_name == "resnet-18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Identity()  # replace last layer to get representation
        else:
            assert model_path is not None
            model = load_model_from_ckpt(
                model_name, str(model_path), finetuner_ckpt_path=None
            )

        self._eventually_freeze_model(model)

        return model, repr_dim

    def _get_predictor(
        self,
        task: Literal["multiclass", "multilabel"],
        input_dim: int,
        num_classes: int,
    ) -> LinearProbingBase:
        if task == "multiclass":
            return MulticlassLinear(input_dim, num_classes)
        elif task == "multilabel":
            return MultilabelLinear(input_dim, num_classes)
        else:
            raise ValueError(f"Invalid task for predictor: {task}")

    def forward(self, x: Tensor) -> Tensor:
        z = self.forward_repr(x)

        if self.norm_repr:
            z = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        preds = self.predictor(z)
        if len(preds.shape) == 3:
            preds = preds.mean(dim=1)

        return preds

    def forward_repr(self, x: Tensor):
        self._eventually_freeze_model(self.encoder)

        model_out = self.encoder(x)
        if isinstance(model_out, tuple):
            _, z, _, _, _ = model_out
            z = torch.swapaxes(z, 0, 1)
        else:
            z = model_out

        return z

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = self.shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        self.shared_step(batch, "test")

    def shared_step(self, batch: tuple[Tensor, Tensor], split: str) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.predictor.loss(input=logits, target=y)

        metrics = self.metrics[f"{split}_metrics"]
        metrics(preds=logits, target=y)
        self.log_dict(metrics)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.hparams.optim_kwargs,
        )

        if self.hparams.scheduler_kwargs is None:
            return {"optimizer": optimizer}

        scheduler = MultiStepLR(
            optimizer=optimizer, **self.hparams.scheduler_kwargs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _eventually_freeze_model(self, model: nn.Module):
        # implements own instead of lightning freeze to be general to any nn.Module
        if self.frozen_encoder:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()


class PreEmbeddedLinearProbingClassifier(LinearProbingClassifier):
    """Version of classifier working on precomputed representations."""

    def _get_encoder(
        self, model_name: str, model_path: Path | None
    ) -> tuple[nn.Module, int]:
        repr_dim = 512
        if model_name == "resnet-50":
            repr_dim = 2048
        return nn.Identity(), repr_dim

    def forward_repr(self, x: Tensor):
        return x
