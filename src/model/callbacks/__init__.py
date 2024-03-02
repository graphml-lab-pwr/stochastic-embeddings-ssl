from pathlib import Path
from typing import Literal

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig

from .online_finetuner import OnlineFineTuner


def get_callbacks(
    cfg: DictConfig,
    output_path: Path,
    train_phase: Literal["train", "finetune"],
) -> list[Callback]:
    callbacks: list[Callback] = []
    task_conf = getattr(cfg, train_phase)

    if "checkpoint" in task_conf.callbacks:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_path / "checkpoints",
                **task_conf.callbacks.checkpoint,
            )
        )

    if "early_stopping" in task_conf.callbacks:
        callbacks.append(EarlyStopping(**task_conf.callbacks.early_stopping))

    if "online_finetuner" in task_conf.callbacks:
        callbacks.append(
            OnlineFineTuner(
                encoder_output_dim=task_conf.kwargs.latent_dim,
                num_classes=cfg.dataset.num_classes,
                **task_conf.callbacks.online_finetuner,
            )
        )

    if "learning_rate_monitor" in task_conf.callbacks:
        callbacks.append(
            LearningRateMonitor(**task_conf.callbacks.learning_rate_monitor)
        )

    return callbacks
