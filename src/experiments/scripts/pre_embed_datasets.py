import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from src.data.datamodule.transfer_learning import (
    get_transfer_learning_datamodule,
)
from src.task.evaluation import LinearProbingClassifier


def main(
    experiment_dir: Optional[Path] = typer.Option(None),
    pre_trained_model: Optional[str] = typer.Option(None),
    config_path: Path = typer.Option(...),
    target_dir: Path = typer.Option(
        ..., help="Dir should include name of the model and dataset"
    ),
):
    assert (experiment_dir is not None) ^ (pre_trained_model is not None)
    target_dir.mkdir(exist_ok=True, parents=True)

    config = _load_config(config_path)
    pl.seed_everything(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = int(os.getenv("NUM_WORKERS", 0))

    if experiment_dir:
        ckpt_cfg = _load_config(experiment_dir / "hydra_config.yaml")
        model_name = ckpt_cfg["train"]["name"]
        mc_samples = ckpt_cfg["train"].get("mc_samples", 1)

        # for z representations does not use mc_samples
        if model_name.endswith("ssl_bayes_z"):
            mc_samples = 1

        model_path = experiment_dir / "checkpoints" / "model.ckpt"
    else:
        model_name = pre_trained_model
        mc_samples = 1
        model_path = None

    dm = get_transfer_learning_datamodule(
        dataset_dir=config["dataset_dir"],
        dataset=config["dataset"],
        batch_size=config["batch_size"],
        num_workers=num_workers,
    )
    dm.setup("train")

    transfer_clf = LinearProbingClassifier(
        model_name=model_name,
        model_path=model_path,
        optim_kwargs=config["optim_kwargs"],
        max_epochs=config["max_epochs"],
        scheduler_kwargs=config["scheduler_kwargs"],
        num_classes=config["num_classes"],
        task=config["task"],
        frozen_encoder=True,
    )

    dataloaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    transfer_clf = transfer_clf.to(device)
    for split_name, split_data_loader in dataloaders.items():
        split_file = target_dir / f"{split_name}.pt"
        if split_data_loader is None:
            continue

        dataset_size = len(split_data_loader.dataset)  # type: ignore[arg-type]
        representations = torch.empty(
            dataset_size, mc_samples, transfer_clf.repr_dim, dtype=torch.float
        )
        representations = representations.squeeze(1)
        labels = torch.empty(dataset_size, dtype=torch.long)

        transfer_clf.eval()
        with torch.no_grad():
            for i, (x, y) in tqdm(
                enumerate(split_data_loader),
                desc=split_name,
                total=len(split_data_loader),
            ):
                x = x.to(device)
                z = transfer_clf.forward_repr(x).cpu()
                z = z.squeeze(1)
                start_idx = i * config["batch_size"]
                end_idx = start_idx + config["batch_size"]
                representations[start_idx:end_idx] = z
                labels[start_idx:end_idx] = y

        save_ds = TensorDataset(representations, labels)
        torch.save(save_ds, split_file)


def _load_config(path: Path) -> dict:
    raw_cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(cfg, dict)

    return cfg


if __name__ == "__main__":
    typer.run(main)
