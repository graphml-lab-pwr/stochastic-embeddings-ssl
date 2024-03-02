from typing import Any, Literal, Optional

from torchvision.datasets import CIFAR10

from src.data.datamodule.base import BaseDataModule


class CIFAR10DataModule(BaseDataModule[CIFAR10]):
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
        **kwargs: Any
    ):
        super().__init__(
            data_dir,
            task,
            val_split,
            batch_size,
            shuffle,
            num_workers,
            drop_last,
            persistent_workers,
            input_height,
            seed,
            pin_memory,
            **kwargs
        )

    def prepare_data(self) -> None:
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()

        if stage == "fit" or stage is None:
            dataset_train = CIFAR10(
                self.data_dir, train=True, transform=self.train_transform
            )
            dataset_val = CIFAR10(
                self.data_dir, train=True, transform=self.val_transform
            )
            self.train_ds = self._split_dataset(dataset_train)
            self.val_ds = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            self.test_ds = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

    @property
    def mean(self) -> list[float]:
        """
        Credits https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py.
        Mean: train_data.mean(axis=(0,1,2)) / 255;
        """
        return [x / 255.0 for x in [125.3, 123.0, 113.9]]

    @property
    def std(self) -> list[float]:
        """
        Credits https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py.
        std: train_data.std(axis=(0,1,2)) / 255.
        """
        return [x / 255.0 for x in [63.0, 62.1, 66.7]]
