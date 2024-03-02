from collections import ChainMap

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.model.callbacks import get_callbacks
from src.task import get_model_cls
from src.utils.loggers import LightningLoggingConfig


def finetune(cfg: DictConfig, ckpt_path: str, run_id: str):
    if cfg.logging.wandb:
        cfg.logging.wandb_logger_kwargs.id = run_id
        cfg.logging.wandb_logger_kwargs.resume = True
    logging_config = LightningLoggingConfig.from_flags(**cfg.logging)

    output_path = cfg.output_path / "finetune"
    output_path.mkdir()

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup(stage="fit")

    model = get_model_cls(cfg).load_from_checkpoint(ckpt_path)
    finetuner = hydra.utils.instantiate(cfg.finetune.kwargs, pl_module=model)

    trainer = pl.Trainer(
        default_root_dir=str(output_path),
        callbacks=get_callbacks(cfg, output_path, train_phase="finetune"),
        logger=logging_config.get_lightning_loggers(output_path),
        **cfg.finetune.trainer,
    )
    trainer.fit(datamodule=datamodule, model=finetuner)
    val_metrics = trainer.validate(datamodule=datamodule, model=finetuner)
    test_metrics = trainer.test(datamodule=datamodule, model=finetuner)

    output = {"run_id": run_id, "metrics": dict(ChainMap(*test_metrics))}

    if "optimized_metric" in cfg:  # hparams search
        output["val_metrics"] = dict(ChainMap(*val_metrics))

    if "checkpoint" in cfg.finetune.callbacks:
        assert trainer.checkpoint_callback is not None
        assert hasattr(trainer.checkpoint_callback, "best_model_path")
        output["ckpt_path"] = trainer.checkpoint_callback.best_model_path
        output["pl_logger"] = finetuner.get_wandb_logger()  # type: ignore

    return output
