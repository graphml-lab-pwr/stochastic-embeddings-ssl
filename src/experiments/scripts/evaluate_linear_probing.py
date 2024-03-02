import json
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.data.datamodule.transfer_learning import (
    PrecomputedRepresentationsDatamodule,
    get_transfer_learning_datamodule,
)
from src.task.evaluation import (
    LinearProbingClassifier,
    PreEmbeddedLinearProbingClassifier,
)

torch.set_float32_matmul_precision("high")


def main(
    backbone_experiment_dir: Optional[Path] = typer.Option(None),
    pre_trained_model: Optional[str] = typer.Option(None),
    config_path: Path = typer.Option(...),
    pre_embedded_root_dir: Path = typer.Option(None),
    test_mode: bool = typer.Option(False),
):
    if test_mode:
        print("Evaluating in TEST_MODE - no validation splits are used!")

    assert (backbone_experiment_dir is not None) ^ (
        pre_trained_model is not None
    )

    config = _load_config(config_path)
    pl.seed_everything(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = int(os.getenv("NUM_WORKERS", 0))

    if backbone_experiment_dir:
        ckpt_cfg = _load_config(backbone_experiment_dir / "hydra_config.yaml")
        pre_train_name = ckpt_cfg["train"]["name"]
        model_name = backbone_experiment_dir.name
        model_path = backbone_experiment_dir / "checkpoints" / "model.ckpt"
    else:
        pre_train_name = pre_trained_model
        model_name = pre_train_name
        model_path = None

    if pre_embedded_root_dir is None:
        dm = get_transfer_learning_datamodule(
            dataset_dir=config["dataset_dir"],
            dataset=config["dataset"],
            batch_size=config["batch_size"],
            num_workers=num_workers,
        )

        transfer_clf = LinearProbingClassifier(
            model_name=pre_train_name,
            model_path=model_path,
            optim_kwargs=config["optim_kwargs"],
            max_epochs=config["max_epochs"],
            scheduler_kwargs=config["scheduler_kwargs"],
            num_classes=config["num_classes"],
            task=config["task"],
            frozen_encoder=True,
            norm_repr=config["norm_repr"],
        )
    else:
        *_, cached_dataset_name, cached_model_name = pre_embedded_root_dir.parts
        assert cached_dataset_name == config["dataset"]
        assert cached_model_name == model_name
        dm = PrecomputedRepresentationsDatamodule(
            data_dir=pre_embedded_root_dir,
            batch_size=config["batch_size"],
            num_workers=num_workers,
            task=config["task"],
            num_classes=config["num_classes"],
            test_mode=test_mode,
        )
        transfer_clf = PreEmbeddedLinearProbingClassifier(
            model_name=pre_train_name,
            model_path=model_path,
            optim_kwargs=config["optim_kwargs"],
            max_epochs=config["max_epochs"],
            scheduler_kwargs=config["scheduler_kwargs"],
            num_classes=config["num_classes"],
            task=config["task"],
            frozen_encoder=True,
            norm_repr=config["norm_repr"],
        )

    experiment_dir = (
        Path(config["experiment_dir"]) / model_name / config["dataset"]
    )

    callbacks: list[pl.Callback] = [
        LearningRateMonitor(logging_interval="step")
    ]
    logger = pl.loggers.TensorBoardLogger(save_dir=experiment_dir)
    if not test_mode:
        callbacks.append(
            ModelCheckpoint(
                monitor="val/accuracy", mode="max", auto_insert_metric_name=True
            )
        )

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=config["max_epochs"],
        logger=logger,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        limit_val_batches=0 if test_mode else 1.0,
    )

    trainer.fit(transfer_clf, datamodule=dm)

    metrics, *_ = trainer.test(transfer_clf, datamodule=dm, ckpt_path="best")
    assert trainer.log_dir is not None
    metrics_path = Path(trainer.log_dir) / "metrics.json"
    with metrics_path.open("w") as file:
        json.dump(metrics, file)


def _load_config(path: Path) -> dict:
    raw_cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(cfg, dict)

    return cfg


if __name__ == "__main__":
    typer.run(main)
