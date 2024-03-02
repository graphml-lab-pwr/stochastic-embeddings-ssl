from typing import Any, Literal

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.task.classification import BaseClassification


class SemiSupervised(BaseClassification):
    def __init__(
        self,
        pl_module: pl.LightningModule,
        finetune_model: Literal["simple_mlp"],
        num_classes: int,
        learning_rate: float,
        use_mc_samples: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: Literal["cosine", "multistep"],
        scheduler_kwargs: dict[str, Any] | None,
        **kwargs: Any
    ):
        super().__init__(
            model=pl_module,
            finetune_model=finetune_model,
            encoder_out_dim=pl_module.encoder_out_dim,  # type: ignore
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_mc_samples=use_mc_samples,
            **kwargs
        )
        self.save_hyperparameters(ignore=["pl_module"])

    def get_x_y_from_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        return x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert hasattr(self.hparams, "use_mc_samples")

        out = self.model(x)
        if isinstance(out, tuple):
            if self.hparams.use_mc_samples:
                feats = out[1]
            else:
                feats = out[0]
        else:
            feats = out

        if self.hparams.use_mc_samples and len(feats.shape) > 2:
            preds = torch.stack(list(map(self.predictor, feats))).mean(dim=0)
        else:
            preds = self.predictor(feats)

        return preds

    def configure_optimizers(self):
        assert hasattr(self.hparams, "scheduler")

        self.hparams.optimizer.keywords.pop("name")
        optimizer = self.hparams.optimizer(
            [
                {
                    "name": "backbone",
                    "params": self.model.parameters(),
                    "lr": self.hparams.optimizer.keywords["lr"],
                    "weight_decay": self.hparams.optimizer.keywords[
                        "weight_decay"
                    ],
                },
                {
                    "name": "linear",
                    "params": self.predictor.parameters(),
                    "weight_decay": self.hparams.optimizer.keywords[
                        "weight_decay"
                    ],
                    "lr": self.hparams.optimizer.keywords.pop("lr_linear"),
                },
            ],
        )

        if self.hparams.scheduler == "multistep":
            assert hasattr(self.hparams.scheduler_kwargs, "milestones")
            assert hasattr(self.hparams.scheduler_kwargs, "gamma")
            scheduler = {
                "scheduler": MultiStepLR(
                    optimizer=optimizer, **self.hparams.scheduler_kwargs
                ),
            }
        elif self.hparams.scheduler == "cosine":
            assert isinstance(self.trainer.max_epochs, int)
            assert hasattr(self.hparams.scheduler_kwargs, "sch_eta_min_scale")
            T_max = self.trainer.max_epochs
            eta_min = (
                self.learning_rate
                * self.hparams.scheduler_kwargs["sch_eta_min_scale"]
            )
            scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer=optimizer, T_max=T_max, eta_min=eta_min
                ),
            }
        else:
            raise ValueError("Scheduler not recognized")

        return [optimizer], [scheduler]
