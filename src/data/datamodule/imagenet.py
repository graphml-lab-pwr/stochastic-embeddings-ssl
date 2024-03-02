import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.request import urlretrieve

from lightly.transforms import BYOLView1Transform, BYOLView2Transform
from torchvision.datasets import ImageNet

from src.data.datamodule.base import BaseDataModule
from src.data.transforms import (
    BarlowTwinsTransform,
    DatasetTransform,
    FinetuneViewTransform,
    ImageNetSupervisedTransform,
)

start_time: float
logger = logging.getLogger(__name__)


def reporthook(count: int, block_size: float, total_size: float):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) * 0.001
    percent = count * block_size * 100 / total_size
    sys.stdout.write(
        "\r%5.1f%%, %d MB, %d MB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


class ImageNetDataModule(BaseDataModule[ImageNet]):
    IMAGENET_URLS = {
        "val": {
            "filename": "ILSVRC2012_img_val.tar",
            "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        },
        "train": {
            "filename": "ILSVRC2012_img_train.tar",
            "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        },
        "devkit": {
            "filename": "ILSVRC2012_devkit_t12.tar.gz",
            "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        },
    }

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
        resize_height: int,
        seed: int,
        pin_memory: bool,
        **kwargs: Any,
    ):
        self.resize_height = resize_height
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
            **kwargs,
        )

    def prepare_data(self) -> None:
        """If hangs simply use wget command to download the specified files."""
        for split, metadata in self.IMAGENET_URLS.items():
            output_path = Path(self.data_dir) / metadata["filename"]
            if not os.path.exists(output_path) and not os.path.islink(
                output_path
            ):
                output_path.parent.mkdir(parents=True)
                logger.info(f"Downloading {metadata['filename']} file...")
                urlretrieve(metadata["url"], output_path, reporthook)

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()

        if stage == "fit" or stage is None:
            self.train_ds = ImageNet(
                self.data_dir, split="train", transform=self.train_transform
            )
            self.val_ds = ImageNet(
                self.data_dir, split="val", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_ds = ImageNet(
                self.data_dir, split="val", transform=self.test_transform
            )

    @property
    def mean(self) -> list[float]:
        """Credits https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self) -> list[float]:
        """Credits https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2."""
        return [0.229, 0.224, 0.225]

    def get_transform(
        self, stage: Literal["train", "val", "test"]
    ) -> DatasetTransform:
        if self.task_type == "ssl":
            return BarlowTwinsTransform(
                view_1_transform=BYOLView1Transform(),
                view_2_transform=BYOLView2Transform(),
                view_3_transform=FinetuneViewTransform(
                    input_size=self.input_height, normalize=self.normalize
                ),
            )
        else:
            return ImageNetSupervisedTransform(
                train=True if stage == "train" else False,
                resize_height=self.resize_height,
                input_height=self.input_height,
                normalize=self.normalize,
            )


class SubsetImageNetDataModule(ImageNetDataModule):
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = ImageNet(
                self.data_dir, split="train", transform=self.train_transform
            )
            self.val_ds = ImageNet(
                self.data_dir, split="val", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_ds = ImageNet(
                self.data_dir, split="val", transform=self.test_transform
            )
