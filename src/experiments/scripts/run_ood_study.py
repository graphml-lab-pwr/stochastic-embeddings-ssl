import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import typer
from openood.evaluation_api import Evaluator
from openood.evaluation_api.preprocessor import default_preprocessing_dict
from openood.preprocessors.barlowtwins_preprocessor import (
    BarlowTwinsPreprocessor,
)
from openood.utils import Config
from tqdm import tqdm

from src.task import (
    HBayesianSelfSupervisedLearning,
    SelfSupervisedLearning,
    ZBayesianSelfSupervisedLearning,
)
from src.task.finetuning import LastLayerFinetuner
from src.utils.utils import load_model_from_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROB_POSTPROCESSORS = ["sigmamean", "logpprior", "kldivsimk"]


def main(
    runs_dir: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
    data_root_dir: Path = typer.Option(...),
    in_dataset: str = typer.Option(...),
    postprocessor_name: str = typer.Option(...),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_cifar10_runs(runs_dir)

    for run in tqdm(runs):
        logger.info(f" Processing {run['run_name']}...")
        run_file = output_dir / f"{run['run_name']}.json"

        if run_file.is_file():
            logger.info(f" Run ({run['run_name']}) already computed...")
            with open(run_file, "r") as f:
                loaded_run = json.load(f)
            run["ood_scores"] = loaded_run["ood_scores"]
        else:
            model = load_model_from_ckpt(
                run["model_name"],
                model_ckpt_path=run["model_path"],
                finetuner_ckpt_path=run["finetuner_path"],
            )
            if (
                run["model_name"] == "ssl"
                and postprocessor_name in PROB_POSTPROCESSORS
            ):
                logger.info(
                    f"Cannot compute OOD scores for run ({run['id']}) due to model={run['model_name']} and prob-based detector..."
                )
                run["ood_scores"] = None
            else:
                if postprocessor_name in PROB_POSTPROCESSORS:
                    config = Config(
                        **{"dataset": default_preprocessing_dict[in_dataset]}
                    )
                    preprocessor = BarlowTwinsPreprocessor(config)
                else:
                    preprocessor = None
                evaluator = Evaluator(
                    net=OpenOODWrapper(model),  # type: ignore
                    id_name=in_dataset,
                    data_root=str(data_root_dir),
                    config_root=None,  # type: ignore
                    preprocessor=preprocessor,
                    postprocessor_name=postprocessor_name,
                    postprocessor=None,
                    batch_size=200,
                    shuffle=False,
                    num_workers=2,
                )
                metrics = evaluator.eval_ood(fsood=False)
                run["ood_scores"] = get_dict_from_metrics(metrics)
            with open(run_file, "w") as f:
                json.dump(run, f)

    # dump all runs
    run_file = output_dir / "runs.json"
    run_file.unlink(missing_ok=True)  # remove if exists
    with open(run_file, "w") as f:
        json.dump(runs, f)


def load_cifar10_runs(runs_dir: str | Path):
    runs = []
    runs_names = [f for f in os.listdir(str(runs_dir)) if not f.startswith(".")]
    for run_name in runs_names:
        runs.append(
            {
                "run_name": run_name,
                "model_path": str(
                    Path(runs_dir) / run_name / "checkpoints" / "model.ckpt"
                ),
                "finetuner_path": str(
                    Path(runs_dir) / run_name / "finetune" / "checkpoints" / "last.ckpt"
                ),
                "model_name": extract_model_name(run_name),
            }
        )
    return runs


def extract_model_name(run_name: str):
    if run_name.startswith("ssl_bayes_h"):
        return "ssl_bayes_h"
    elif run_name.startswith("ssl_bayes_z"):
        return "ssl_bayes_z"
    else:
        return "ssl"


def get_dict_from_metrics(metrics):
    return (
        metrics.reset_index()
        .rename(columns={"index": "Dataset"})
        .to_dict(orient="records")
    )


class OpenOODWrapper:
    def __init__(self, model: LastLayerFinetuner):
        self.model = model.model
        self.predictor = model.predictor

    def __call__(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
        return_feature_list: bool = False,
        return_embeddings: bool = False,
        return_dist: bool = False,
        *args: Any,
        **kwarg: Any,
    ):
        model_output = self.model(x)

        if isinstance(model_output, tuple):  # [ssl_bayes_h]
            feats, h, mu, log_var, q = model_output
            embeddings = self.model.project(h).mean(dim=0)
            preds = torch.stack(list(map(self.predictor, h))).mean(dim=0)
            dist = (mu, log_var)
        else:  # [ssl, ssl_bayes_z]
            feats = model_output
            projector_output = self.model.project(feats)
            if isinstance(projector_output, tuple):  # [ssl_bayes_z]
                embeddings, z, mu, log_var, q = projector_output
                dist = (mu, log_var)
            else:  # [ssl]
                embeddings = projector_output
                dist = None
            preds = self.predictor(feats)

        extras = []
        if return_feature:
            extras.append(feats)
        if return_feature_list:
            extras.append(self.get_feature_list(x))
        if return_embeddings:
            extras.append(embeddings)
        if return_dist:
            extras.append(dist)

        if not extras:
            return preds
        else:
            return [preds] + extras

    def get_feature_list(self, x: torch.Tensor):
        """This is implementation for ResNet-18 encoder -- won't work properly for other encoders!"""
        enc = self.model.encoder

        feature1 = F.relu(enc.bn1(self.model.encoder.conv1(x.cuda())))
        feature2 = enc.layer1(feature1)
        feature3 = enc.layer2(feature2)
        feature4 = enc.layer3(feature3)
        feature5 = enc.layer4(feature4)
        feature5 = enc.avgpool(feature5)

        return [feature1, feature2, feature3, feature4, feature5]

    def eval(self):
        self.model.eval()
        self.predictor.eval()

    @property
    def stochastic_space(self) -> str | None:
        if isinstance(self.model, ZBayesianSelfSupervisedLearning):
            return "Z"
        elif isinstance(self.model, HBayesianSelfSupervisedLearning):
            return "H"
        elif isinstance(self.model, SelfSupervisedLearning):
            return None
        else:
            raise ValueError(
                "Wrong `model_type` given as a model; not recognized!"
            )


if __name__ == "__main__":
    typer.run(main)
