import os
from enum import Enum
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__)).parent.absolute()
PROJECT_DIR = ROOT_DIR / "bayes_ssl"

DATA_PATH = ROOT_DIR / "data"
DATASET_PATH = DATA_PATH / "datasets"
MODEL_PATH = DATA_PATH / "models"

RESULTS_DIR = DATA_PATH / "results"
CONFIG_DIR = PROJECT_DIR / "experiments" / "configs"

DVC_METRICS_PATH = RESULTS_DIR / "metrics.json"


class TrainPhase(str, Enum):
    train = "train"
    finetune = "finetune"
    joint = "joint"
