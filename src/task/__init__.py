from typing import Type

from hydra._internal.utils import _locate
from omegaconf import DictConfig

from src.task.bayesian_ssl import (
    HBayesianSelfSupervisedLearning,
    ZBayesianSelfSupervisedLearning,
)
from src.task.classification import Classification
from src.task.ssl import SelfSupervisedLearning

_ = (
    Classification,
    HBayesianSelfSupervisedLearning,
    ZBayesianSelfSupervisedLearning,
    SelfSupervisedLearning,
)


def get_model_cls(cfg: DictConfig):
    return _locate(cfg.train.kwargs["_target_"])


def get_model_cls_from_name(
    model_name: str,
) -> (
    Type[ZBayesianSelfSupervisedLearning]
    | Type[HBayesianSelfSupervisedLearning]
    | Type[SelfSupervisedLearning]
):
    if model_name == "ssl_bayes_z":
        return ZBayesianSelfSupervisedLearning
    elif model_name == "ssl_bayes_h":
        return HBayesianSelfSupervisedLearning
    elif model_name == "ssl":
        return SelfSupervisedLearning
    else:
        raise ValueError("Wrong `model_name` argument given! (not recognized)")
