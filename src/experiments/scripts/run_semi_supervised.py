import json
import logging
import multiprocessing as mp
from pathlib import Path

import hydra
import wandb
from lightning_lite import seed_everything
from omegaconf import DictConfig, OmegaConf

from src.experiments.train import train
from src.utils.utils import save_eval_results

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf", config_name="semi_supervised", version_base=None
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    cfg.run_path = Path(cfg.run_path)
    cfg.output_path = Path(cfg.output_path)
    cfg.output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"\nCurrent run's config:\n{OmegaConf.to_yaml(cfg)}\n")

    with open(cfg.run_path / "results.json", "r") as f:
        run = json.load(f)

    output = train(cfg=cfg, run=run)
    save_eval_results(cfg=cfg, train_output=output, finetune_output=None)

    if "logging" in cfg and cfg.logging.wandb:
        if (
            output is not None
            and "pl_logger" in output
            and output["pl_logger"] is not None
        ):
            output["pl_logger"].finalize("success")
        logger.info("Finishing WandB run...")
        wandb.finish()


# if statement due to https://stackoverflow.com/questions/70890187/referring-to-hydras-conf-directory-from-a-python-sub-sub-sub-directory-module
if __name__ == "__main__":
    # due to very slow dataloading for imagenet
    # https://github.com/pytorch/pytorch/issues/102269
    mp.set_start_method("spawn")
    main()
