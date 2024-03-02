# Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py

import logging
from importlib.util import find_spec
from typing import Any, Callable

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            output = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            logger.exception(f"Detected following exception:\n{ex}")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if "logging" in cfg and cfg.logging.wandb:
                    logger.info("Closing wandb!")
                    wandb.finish()

        return output

    return wrap
