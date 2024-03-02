import json
from typing import Any, Type

from omegaconf import DictConfig, OmegaConf

from src.task import (
    HBayesianSelfSupervisedLearning,
    SelfSupervisedLearning,
    ZBayesianSelfSupervisedLearning,
)
from src.task.finetuning import LastLayerFinetuner
from src.utils.loggers import get_logger

logger = get_logger(__name__)


def save_eval_results(
    cfg: DictConfig,
    train_output: dict[str, Any],
    finetune_output: dict[str, Any] | None,
):
    results = {
        "metrics": train_output["metrics"],
        "model_ckpt_path": train_output["ckpt_path"],
        "metadata": {
            "dataset": cfg.dataset.name,
            **OmegaConf.to_container(cfg.train.kwargs, resolve=True),  # type: ignore
        },
    }
    if "run_id" in train_output:
        results["metadata"]["run_id"] = train_output["run_id"]
    if finetune_output is not None:
        results["offline_metrics"] = finetune_output["metrics"]
        results["finetuner_ckpt_path"] = finetune_output["ckpt_path"]

    with open(cfg.output_path / "results.json", "w") as file:
        json.dump(results, file, indent="\t")


def get_metric_value(metric_dict: dict, metric_name: str) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    From https://github.com/ashleve/lightning-hydra-template/blob/5cdced33be5b87affb8a7db94ef1f8c40a356cc0/src/utils/utils.py
    """
    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    logger.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def load_model_from_ckpt(
    model_name: str, model_ckpt_path: str, finetuner_ckpt_path: str | None
):
    model_cls: (
        Type[HBayesianSelfSupervisedLearning]
        | Type[ZBayesianSelfSupervisedLearning]
        | Type[SelfSupervisedLearning]
    )
    if model_name == "ssl_bayes_z":
        model_cls = ZBayesianSelfSupervisedLearning
    elif model_name == "ssl_bayes_h":
        model_cls = HBayesianSelfSupervisedLearning
    elif model_name == "ssl":
        model_cls = SelfSupervisedLearning
    else:
        raise ValueError("Wrong `model_name` argument given! (not recognized)")
    model = model_cls.load_from_checkpoint(model_ckpt_path).eval()
    use_mean_for_repr = (
        model.hparams.use_mean_for_repr
        if hasattr(model.hparams, "use_mean_for_repr")
        else False
    )
    if finetuner_ckpt_path is not None:
        finetuner = LastLayerFinetuner.load_from_checkpoint(
            finetuner_ckpt_path,
            pl_module=model,
            use_mean_for_repr=use_mean_for_repr,
            no_grad=False,
            # add missing arguments
            offline_learning_rate=1e-3,  # each experiment has the same lr for finetuning
            finetune_model="simple_mlp",  # is actually 1 linear layer
        ).eval()
        return finetuner
    else:
        return model
