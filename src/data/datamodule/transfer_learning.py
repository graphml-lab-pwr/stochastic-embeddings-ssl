from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import Flowers102, ImageNet

from src.data.dataset.sun397 import SUN397Split
from src.data.transforms import FinetuneViewTransform


def get_transfer_learning_datamodule(
    dataset_dir: Path,
    dataset: str,
    batch_size: int,
    num_workers: int,
) -> "BaseLinearProbingDatamodule":
    if dataset == "SUN397":
        return SUN397Datamodule(
            data_dir=dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "flowers-102":
        return Flowers102Datamodule(
            data_dir=dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "imagenet":
        return ImagenetDatamodule(
            data_dir=dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset}")


def _split_dataset(
    base_dataset: Dataset,
    test_frac: float,
    stratify: torch.Tensor | None = None,
) -> tuple[Subset, Subset]:
    train_idx, test_idx = train_test_split(
        torch.arange(len(base_dataset)),  # type: ignore[arg-type]
        stratify=stratify,
        test_size=test_frac,
    )
    return Subset(base_dataset, train_idx), Subset(base_dataset, test_idx)


class BaseLinearProbingDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int,
        task: Literal["multiclass", "multilabel"],
        num_classes: int,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0

        self.train_ds: Dataset | None
        self.val_ds: Dataset | None
        self.test_ds: Dataset | None

        self.task = task
        self.num_classes = num_classes

    @property
    def tranforms(self):
        # NOTE: the parameter holds only for ImageNet-pretrained models
        return FinetuneViewTransform(
            input_size=224,
            normalize={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._create_dataloader(dataset=self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        assert self.val_ds is not None
        return self._create_dataloader(dataset=self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        assert self.test_ds is not None
        return self._create_dataloader(dataset=self.test_ds, shuffle=False)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )


class SUN397Datamodule(BaseLinearProbingDatamodule):
    def __init__(self, data_dir: Path, batch_size: int, num_workers: int):
        super().__init__(data_dir, batch_size, num_workers, "multiclass", 397)

    def prepare_data(self) -> None:
        SUN397Split(
            self.data_dir,
            partition_id=1,
            split="train",
            download=True,
            transform=self.tranforms,
        )
        SUN397Split(
            self.data_dir,
            partition_id=1,
            split="test",
            download=True,
            transform=self.tranforms,
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = SUN397Split(
            self.data_dir,
            partition_id=1,
            split="train",
            transform=self.tranforms,
        )
        self.test_ds = SUN397Split(
            self.data_dir,
            partition_id=1,
            split="test",
            transform=self.tranforms,
        )

    def val_dataloader(self) -> DataLoader | None:
        return None


class Flowers102Datamodule(BaseLinearProbingDatamodule):
    def __init__(self, data_dir: Path, batch_size: int, num_workers: int):
        super().__init__(data_dir, batch_size, num_workers, "multiclass", 102)
        self.prepare_data()

    def prepare_data(self) -> None:
        Flowers102(
            self.data_dir,
            split="train",
            download=True,
            transform=self.tranforms,
        )
        Flowers102(
            self.data_dir, split="val", download=True, transform=self.tranforms
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = Flowers102(
            self.data_dir, split="train", transform=self.tranforms
        )
        self.val_ds = Flowers102(
            self.data_dir, split="val", transform=self.tranforms
        )

    def test_dataloader(self) -> DataLoader | None:
        return None


class ImagenetDatamodule(BaseLinearProbingDatamodule):
    DEFAULT_VAL_FRAC = 0.2

    def __init__(self, data_dir: Path, batch_size: int, num_workers: int):
        super().__init__(data_dir, batch_size, num_workers, "multiclass", 1_000)
        self.prepare_data()

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = ImageNet(
            self.data_dir, split="train", transform=self.tranforms
        )
        self.val_ds = ImageNet(
            self.data_dir, split="val", transform=self.tranforms
        )

    def test_dataloader(self) -> DataLoader | None:
        return None


class PrecomputedRepresentationsDatamodule(BaseLinearProbingDatamodule):
    DEFAULT_VAL_FRAC = 0.2

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int,
        task: Literal["multiclass", "multilabel"],
        num_classes: int,
        test_mode: bool,
    ):
        super().__init__(data_dir, batch_size, num_workers, task, num_classes)
        self.test_mode = test_mode

    @property
    def tranforms(self):
        return None

    def setup(self, stage: str) -> None:
        self.train_ds = self._load_dataset("train")
        self.val_ds = self._load_dataset("val")
        self.test_ds = self._load_dataset("test")

        # swap test and validation set to compute final results on validation
        if self.test_ds is None:
            assert self.val_ds is not None
            self.test_ds = self.val_ds
            self.val_ds = None

        # in test_mode we skip validation loop and only evaluate test set at the end of training
        if self.test_mode:
            return

        # without test_mode we perform spit on train set to have val_ds and run validation loop
        if self.val_ds is None:
            assert self.train_ds is not None
            _, train_labels = self.train_ds.tensors
            self.train_ds, self.val_ds = _split_dataset(
                self.train_ds, self.DEFAULT_VAL_FRAC, stratify=train_labels
            )

            assert self.train_ds is not None
            assert self.val_ds is not None
            assert self.test_ds is not None

    def _load_dataset(self, split: str) -> TensorDataset | None:
        split_path = Path(self.data_dir) / f"{split}.pt"

        try:
            return torch.load(split_path)
        except FileNotFoundError:
            return None
