import abc
from math import ceil
from typing import Any, Generic, Literal, Optional, TypeVar

import torch
import torch.utils.data as data
import torchvision.transforms as T
from lightly.transforms import BYOLView1Transform, BYOLView2Transform
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split

from src.data.transforms import (
    BarlowTwinsTransform,
    DatasetTransform,
    FinetuneViewTransform,
    SupervisedTransform,
)

Dataset = TypeVar("Dataset", bound=data.Dataset)


class BaseDataModule(LightningDataModule, abc.ABC, Generic[Dataset]):
    def __init__(
        self,
        data_dir: str,
        task: Literal[
            "supervised",
            "semi_supervised",
            "linear_classification",
            "ssl",
            "ssl_bayes_h",
            "ssl_bayes_z",
            "ssl_bayes_h_and_z",
        ],
        val_split: float,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        persistent_workers: bool,
        input_height: int,
        seed: int,
        pin_memory: bool,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_ds: Dataset | Subset[Dataset]
        self.val_ds: Dataset | Subset[Dataset]
        self.test_ds: Dataset | Subset[Dataset]

        self.data_dir = data_dir
        self.task = task
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.input_height = input_height
        self.seed = seed
        self.pin_memory = pin_memory

    @property
    def task_type(
        self,
    ) -> Literal[
        "ssl", "supervised", "semi_supervised", "linear_classification"
    ]:
        if self.task in [
            "ssl",
            "ssl_bayes_h",
            "ssl_bayes_h_and_z",
            "ssl_bayes_z",
            "ssl_bayes_hierarchical",
        ]:
            return "ssl"
        elif self.task == "supervised":
            return "supervised"
        elif self.task == "semi_supervised":
            return "semi_supervised"
        elif self.task == "linear_classification":
            return "linear_classification"
        else:
            raise ValueError("Cannot infer `task_type` due to wrong task name!")

    @property
    def dataloader_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }

    @property
    def num_training_samples(self) -> int:
        if not hasattr(self, "train_ds"):
            raise AttributeError(
                "Train dataset (`train_ds`) not available. Call `setup()` method."
            )
        return len(self.train_ds)  # type: ignore

    @property
    def train_iters_per_epoch(self) -> int:
        if not hasattr(self, "train_ds"):
            raise AttributeError(
                "Train dataset (`train_ds`) not available. Call `setup()` method."
            )
        if self.drop_last:
            return self.num_training_samples // self.batch_size
        else:
            return ceil(self.num_training_samples / self.batch_size)

    def _split_dataset(
        self, dataset: Dataset, train: bool = True
    ) -> Subset[Dataset]:
        """From https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py"""
        len_dataset = len(dataset)  # type: ignore
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )
        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> list[int]:
        """From https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py"""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    @abc.abstractmethod
    def prepare_data(self) -> None:
        pass

    @abc.abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.train_ds, shuffle=self.shuffle, **self.dataloader_kwargs
        )

    def val_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(self.val_ds, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(self.test_ds, shuffle=False, **self.dataloader_kwargs)

    @property
    def train_transform(self) -> DatasetTransform:
        return self.get_transform(stage="train")

    @property
    def val_transform(self) -> DatasetTransform:
        return self.get_transform(stage="val")

    @property
    def test_transform(self) -> DatasetTransform:
        return self.get_transform(stage="test")

    def get_transform(
        self, stage: Literal["train", "val", "test"]
    ) -> DatasetTransform:
        if self.task_type == "ssl":
            return BarlowTwinsTransform(
                view_1_transform=BYOLView1Transform(
                    input_size=self.input_height,
                    gaussian_blur=0.0,
                    normalize=self.normalize,
                ),
                view_2_transform=BYOLView2Transform(
                    input_size=self.input_height,
                    gaussian_blur=0.0,
                    normalize=self.normalize,
                ),
                view_3_transform=FinetuneViewTransform(
                    input_size=self.input_height, normalize=self.normalize
                ),
            )
        else:
            return SupervisedTransform(
                train=True if stage == "train" else False,
                input_height=self.input_height,
                normalize=self.normalize,
            )

    @property
    @abc.abstractmethod
    def mean(self) -> float | list[float]:
        pass

    @property
    @abc.abstractmethod
    def std(self) -> float | list[float]:
        pass

    @property
    def normalize(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def normalize_fn(self) -> torch.nn.Module:
        return T.Normalize(mean=self.mean, std=self.std)
