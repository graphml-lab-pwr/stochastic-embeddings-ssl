from typing import Any, Literal, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.functional import accuracy


class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # noinspection PyTypeChecker
        self.task: Literal["binary", "multiclass", "multilabel"] = (
            "binary" if num_classes == 2 else "multiclass"
        )

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        assert callable(pl_module.init_downstream_model)
        pl_module.init_downstream_model()
        self.optimizer = torch.optim.Adam(
            pl_module.online_finetuner.parameters(), lr=self.learning_rate
        )

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss, acc = self.shared_step(pl_module, batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("train_online_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("train_online_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, acc = self.shared_step(pl_module, batch)
        pl_module.log(
            "val_online_acc", acc, on_step=False, on_epoch=True, sync_dist=True
        )
        pl_module.log(
            "val_online_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def shared_step(
        self,
        pl_module: "pl.LightningModule",
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert callable(pl_module.online_finetuner)
        assert isinstance(pl_module.device, torch.device)
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            out = pl_module(x)

        if isinstance(out, tuple):
            feats, *_ = pl_module(x)
        else:
            feats = out
        feats = feats.detach()

        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = accuracy(
            F.softmax(preds, dim=1),
            y,
            task=self.task,
            num_classes=self.num_classes,
        )

        return loss, acc
