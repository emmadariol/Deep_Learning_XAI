"""Dataset and DataLoader utilities for image-classification manifests."""

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
class ManifestSample:
    """One image-classification sample loaded from a CSV manifest."""

    filepath: Path
    label: int
    class_name: str
    split: str
    extra: dict[str, str]


class ImageManifestDataset(Dataset):
    """PyTorch Dataset backed by a manifest with filepath/label/class/split columns.

    Required columns:

    - ``filepath``
    - ``label``
    - ``class_name``
    - ``split``

    Additional columns, such as Oxford Pets ``mask_path`` or ``species``, are preserved
    in ``sample.extra`` but are not required by the AwA2 pipeline.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_dir = self.manifest_path.parent
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' in {self.manifest_path}")

        self.classes = sorted({sample.class_name for sample in self.samples})
        LOGGER.info(
            "Loaded manifest split=%s with %d samples and %d visible classes from %s",
            split,
            len(self.samples),
            len(self.classes),
            self.manifest_path,
        )

    def _resolve_image_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.manifest_dir / path).resolve()

    def _load_samples(self) -> list[ManifestSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        samples: list[ManifestSample] = []
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
                    ManifestSample(
                        filepath=self._resolve_image_path(row["filepath"]),
                        label=int(row["label"]),
                        class_name=row["class_name"],
                        split=row["split"],
                        extra={
                            key: value
                            for key, value in row.items()
                            if key not in {"filepath", "label", "class_name", "split"}
                        },
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


AwA2Dataset = ImageManifestDataset
OxfordPetDataset = ImageManifestDataset


def build_resnet_transforms(train: bool = False) -> transforms.Compose:
    """Return standard ImageNet preprocessing for ResNet fine-tuning."""
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
    """Build train/val/test dataloaders from an image-classification manifest."""
    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = ImageManifestDataset(
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
