"""Data loading utilities for the Animals with Attributes 2 dataset."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AwA2Sample:
    filepath: Path
    label: int
    class_name: str
    split: str


class AwA2Dataset(Dataset):
    """PyTorch Dataset backed by a CSV manifest produced by prepare_awa2.py."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(
                f"No samples found for split='{split}' in {self.manifest_path}"
            )

        self.classes = sorted({sample.class_name for sample in self.samples})
        LOGGER.info(
            "Loaded AwA2 split=%s with %d samples and %d visible classes",
            split,
            len(self.samples),
            len(self.classes),
        )

    def _load_samples(self) -> list[AwA2Sample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        samples: list[AwA2Sample] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"filepath", "label", "class_name", "split"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            for row in reader:
                if row["split"] != self.split:
                    continue
                samples.append(
                    AwA2Sample(
                        filepath=Path(row["filepath"]).expanduser().resolve(),
                        label=int(row["label"]),
                        class_name=row["class_name"],
                        split=row["split"],
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        sample = self.samples[index]
        image = Image.open(sample.filepath).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label, sample.class_name, str(sample.filepath)


def build_resnet_transforms(train: bool = False) -> transforms.Compose:
    """Return standard ImageNet preprocessing for ResNet fine-tuning.

    The train flag is intentionally conservative in Phase 1: no stochastic
    augmentation yet, so normalization checks are identical across splits.
    """
    _ = train
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_debug_transform() -> transforms.Compose:
    """Return preprocessing without normalization for tensor range inspection."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def build_dataloaders(
    manifest_path: str | Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from the AwA2 manifest."""
    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = AwA2Dataset(
            manifest_path=manifest_path,
            split=split,
            transform=build_resnet_transforms(train=split == "train"),
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return dataloaders


def denormalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization and sanity checks."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return batch * std + mean
