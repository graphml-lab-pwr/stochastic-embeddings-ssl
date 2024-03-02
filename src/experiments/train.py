from collections import ChainMap
from typing import Any

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.model.callbacks import get_callbacks
from src.task import get_model_cls_from_name
from src.utils.loggers import LightningLoggingConfig


def train(cfg: DictConfig, run: dict[str, Any]):
    logging_config = LightningLoggingConfig.from_flags(**cfg.logging)

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup(stage="fit")

    ckpt_path = run["model_ckpt_path"]
    model_name = run["metadata"]["model_name"]
    pl_module = get_model_cls_from_name(model_name).load_from_checkpoint(
        ckpt_path
    )
    model = hydra.utils.instantiate(
        cfg.train.kwargs,
        pl_module=pl_module,
        run_name=cfg.run_path.name,
        **get_metadata(run),
    )

    trainer = pl.Trainer(
        default_root_dir=str(cfg.output_path),
        callbacks=get_callbacks(cfg, cfg.output_path, train_phase="train"),
        logger=logging_config.get_lightning_loggers(cfg.output_path),
        **cfg.train.trainer,
    )
    trainer.fit(datamodule=datamodule, model=model)
    test_metrics = trainer.test(datamodule=datamodule, model=model)

    output = {"metrics": dict(ChainMap(*test_metrics))}

    if "checkpoint" in cfg.train.callbacks:
        assert trainer.checkpoint_callback is not None
        assert hasattr(trainer.checkpoint_callback, "best_model_path")
        output["ckpt_path"] = trainer.checkpoint_callback.best_model_path
        output["pl_logger"] = model.get_wandb_logger()  # type: ignore

    return output


def get_metadata(run: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        k: v
        for k, v in run["metadata"].items()
        if k in ["model_name", "cl_loss_name", "beta_scale"]
    }
    metadata["original_run_id"] = run["metadata"]["run_id"]
    return metadata
