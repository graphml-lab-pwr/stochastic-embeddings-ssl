import logging
import multiprocessing as mp
from pathlib import Path

import hydra
import wandb
from lightning_lite import seed_everything
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib.runid import generate_id

from src.experiments.finetune import finetune
from src.experiments.pretrain import pretrain
from src.utils.task_wrapper import task_wrapper
from src.utils.utils import save_eval_results

logger = logging.getLogger(__name__)


@task_wrapper
def run_pipeline(cfg: DictConfig):
    seed_everything(cfg.seed)
    run_id = generate_id()

    logger.info(f"\nGenerated RUN ID: {run_id}\n")
    logger.info(f"\nCurrent run's config:\n{OmegaConf.to_yaml(cfg)}\n")

    cfg.output_path = Path(cfg.output_path)
    cfg.output_path.mkdir(exist_ok=True, parents=True)
    with open(cfg.output_path / "hydra_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if "train" in cfg:
        train_output = pretrain(cfg=cfg, run_id=run_id)
        model_ckpt_path = train_output["ckpt_path"]
        if "finetune" in cfg:
            finetune_output = finetune(
                cfg=cfg, ckpt_path=model_ckpt_path, run_id=run_id
            )
            finetuner_ckpt_path = finetune_output["ckpt_path"]
        else:
            finetune_output = None
            finetuner_ckpt_path = None
        save_eval_results(
            cfg=cfg, train_output=train_output, finetune_output=finetune_output
        )
    else:
        train_output = None
        finetune_output = None
        model_ckpt_path = cfg.model_ckpt_path or None
        finetuner_ckpt_path = cfg.finetuner_ckpt_path or None

    if "logging" in cfg and cfg.logging.wandb:
        if (
            train_output is not None
            and "pl_logger" in train_output
            and train_output["pl_logger"] is not None
        ):
            train_output["pl_logger"].finalize("success")
        if (
            finetune_output is not None
            and "pl_logger" in finetune_output
            and finetune_output["pl_logger"] is not None
        ):
            finetune_output["pl_logger"].finalize("success")
        logger.info("Finishing WandB run...")
        wandb.finish()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    return run_pipeline(cfg)


# if statement due to https://stackoverflow.com/questions/70890187/referring-to-hydras-conf-directory-from-a-python-sub-sub-sub-directory-module
if __name__ == "__main__":
    # due to very slow dataloading for imagenet
    # https://github.com/pytorch/pytorch/issues/102269
    mp.set_start_method("spawn")
    main()
