from collections import ChainMap

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.model.callbacks import get_callbacks
from src.utils.loggers import LightningLoggingConfig


def pretrain(cfg: DictConfig, run_id: str):
    torch.set_float32_matmul_precision("medium")

    if cfg.logging.wandb:
        cfg.logging.wandb_logger_kwargs.id = run_id
    logging_config = LightningLoggingConfig.from_flags(**cfg.logging)

    if "finetune" in cfg:
        output_path = cfg.output_path / "train"
        output_path.mkdir()

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup(stage="fit")

    cfg.train.kwargs.train_iters_per_epoch = datamodule.train_iters_per_epoch
    model = hydra.utils.instantiate(cfg.train.kwargs, seed=cfg.seed)

    trainer = L.Trainer(
        default_root_dir=str(cfg.output_path),
        callbacks=get_callbacks(cfg, cfg.output_path, train_phase="train"),
        logger=logging_config.get_lightning_loggers(cfg.output_path),
        **cfg.train.trainer,
    )
    trainer.fit(datamodule=datamodule, model=model)
    metrics = trainer.validate(datamodule=datamodule, model=model)

    output = {"run_id": run_id, "metrics": dict(ChainMap(*metrics))}
    if "checkpoint" in cfg.train.callbacks:
        assert trainer.checkpoint_callback is not None
        assert hasattr(trainer.checkpoint_callback, "best_model_path")
        output["ckpt_path"] = trainer.checkpoint_callback.best_model_path
        output["pl_logger"] = model.get_wandb_logger()  # type: ignore

    return output
