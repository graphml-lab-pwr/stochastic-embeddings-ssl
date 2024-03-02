from pathlib import Path
from typing import Callable, Literal

from torchvision.datasets import SUN397
from torchvision.datasets.utils import download_and_extract_archive


class SUN397Split(SUN397):
    _PARITITON_URL = (
        "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
    )
    _PARTITION_NAMES = {"train": "Training", "test": "Testing"}

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        partition_id: int,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, download)

        self.partition_id = partition_id
        self.split = split

        # filter to contain inly images from split
        split_imgs = self._load_partition_and_split()
        self._image_files: list[Path] = [
            img_file
            for img_file in self._image_files
            if f"/{img_file.relative_to(self._data_dir)}" in split_imgs
        ]
        self._labels = [
            self.class_to_idx[
                "/".join(path.relative_to(self._data_dir).parts[1:-1])
            ]
            for path in self._image_files
        ]

        assert len(split_imgs) == len(self._image_files)

    def __len__(self) -> int:
        return len(self._image_files)

    def _download(self):
        super()._download()
        partitions_dir = self._data_dir / "Partitions"
        if not partitions_dir.is_dir():
            download_and_extract_archive(
                self._PARITITON_URL, partitions_dir, filename="Partitions.zip"
            )

    def _load_partition_and_split(self) -> set[str]:
        filename = (
            f"{self._PARTITION_NAMES[self.split]}_{self.partition_id:02d}.txt"
        )
        with (self._data_dir / "Partitions" / filename).open() as file:
            split_filenames = file.read().splitlines()
        return set(split_filenames)
