from typing import Any, Literal

import lightning.pytorch as pl
import torch
from torch.distributions import Distribution, MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, CalibrationError, MetricCollection

from src.task.classification import BaseClassification


class LastLayerFinetuner(BaseClassification):
    def __init__(
        self,
        pl_module: pl.LightningModule,
        finetune_model: Literal["simple_mlp"],
        num_classes: int,
        no_grad: bool,
        offline_learning_rate: float,
        offline_weight_decay: float = 1e-4,
        use_mc_samples: bool = False,
        use_mean_for_repr: bool = False,
        **kwargs: Any
    ):
        super().__init__(
            model=pl_module,
            finetune_model=finetune_model,
            encoder_out_dim=pl_module.encoder_out_dim,  # type: ignore
            num_classes=num_classes,
            learning_rate=offline_learning_rate,
            no_grad=no_grad,
            use_mc_samples=use_mc_samples,
            use_mean_for_repr=use_mean_for_repr,
            **kwargs
        )
        self.save_hyperparameters(ignore=["pl_module"])

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
                "offline_acc": acc,  # type: ignore
                "offline_acc_at_5": acc_at_5,  # type: ignore
                "offline_ece": ece,  # type: ignore
            }
        )

    def get_x_y_from_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(self.device)
        y = y.to(self.device)

        return finetune_view, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.train_metrics(preds, labels)
        self.log("train_offline_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.val_metrics(preds, labels)
        self.log("val_offline_loss", loss, batch_size=len(batch), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        _, labels = self.get_x_y_from_batch(batch)
        loss, preds = self._shared_step(batch)
        self.test_metrics(preds, labels)
        self.log("test_offline_loss", loss, on_step=False, on_epoch=True)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x, _ = self.get_x_y_from_batch(batch)
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert hasattr(self.hparams, "use_mc_samples") and hasattr(
            self.hparams, "no_grad"
        )

        feats = self.features(x)
        if self.hparams.no_grad:
            feats = feats.detach()

        if self.hparams.use_mc_samples and len(feats.shape) > 2:
            preds = torch.stack(list(map(self.predictor, feats))).mean(dim=0)
        else:
            preds = self.predictor(feats)

        return preds

    def features(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            hasattr(self.hparams, "use_mc_samples")
            and hasattr(self.hparams, "use_mean_for_repr")
            and hasattr(self.hparams, "no_grad")
        )

        if self.hparams.no_grad:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)

        if isinstance(out, tuple):
            reprs, z, mu, *_ = out
            if self.hparams.use_mc_samples:
                return z
            elif self.hparams.use_mean_for_repr:
                return mu
            else:
                return reprs
        else:
            return out

    def features_ood(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        if len(features.shape) == 3:
            features = torch.mean(features, dim=0)
        return features

    def latent_repr(self, x: torch.Tensor) -> Distribution:
        assert hasattr(self.hparams, "no_grad")
        if self.hparams.no_grad:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)

        if isinstance(out, tuple):
            reprs, z, mu, sigma, q = out
        else:
            raise TypeError("Wrong model output type")

        return MultivariateNormal(mu, torch.diag_embed(sigma))

    def configure_optimizers(self) -> Any:
        assert hasattr(self.hparams, "offline_learning_rate")
        assert hasattr(self.hparams, "offline_weight_decay")
        assert isinstance(self.trainer.max_epochs, int)
        optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.hparams.offline_learning_rate,
            weight_decay=self.hparams.offline_weight_decay,
        )
        T_max = self.trainer.max_epochs
        eta_min = self.hparams.offline_learning_rate * 0.1
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer=optimizer, T_max=T_max, eta_min=eta_min
            ),
        }
        return [optimizer], [scheduler]
