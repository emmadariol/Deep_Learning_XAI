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


def infer_class_map_path(manifest_path: str | Path) -> Path | None:
    """Return the matching class mapping path when it follows project naming."""
    manifest = Path(manifest_path).expanduser().resolve()
    candidates: list[Path] = []

    if manifest.stem.startswith("awa2_manifest"):
        suffix = manifest.stem.removeprefix("awa2_manifest")
        candidates.append(manifest.with_name(f"class_to_idx{suffix}.csv"))

    candidates.append(manifest.with_name("class_to_idx.csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_class_mapping(class_map_path: str | Path) -> dict[str, int]:
    """Load a stable class_name -> label mapping produced by prepare_awa2.py."""
    path = Path(class_map_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Class mapping not found: {path}")

    class_to_idx: dict[str, int] = {}
    seen_labels: set[int] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"class_name", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Class mapping is missing columns: {sorted(missing)}")

        for row in reader:
            class_name = row["class_name"]
            label = int(row["label"])
            if class_name in class_to_idx:
                raise ValueError(f"Duplicate class name in mapping: {class_name}")
            if label in seen_labels:
                raise ValueError(f"Duplicate label in mapping: {label}")
            class_to_idx[class_name] = label
            seen_labels.add(label)

    if not class_to_idx:
        raise ValueError(f"Class mapping is empty: {path}")

    labels = sorted(class_to_idx.values())
    expected = list(range(len(labels)))
    if labels != expected:
        raise ValueError(
            "Class mapping labels must be contiguous and zero-based: "
            f"expected {expected}, found {labels}"
        )

    return dict(sorted(class_to_idx.items(), key=lambda item: item[1]))


class AwA2Dataset(Dataset):
    """PyTorch Dataset backed by a CSV manifest produced by prepare_awa2.py."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable | None = None,
        class_map_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.class_map_path = (
            Path(class_map_path).expanduser().resolve()
            if class_map_path is not None
            else infer_class_map_path(self.manifest_path)
        )
        self.class_to_idx = (
            load_class_mapping(self.class_map_path)
            if self.class_map_path is not None
            else {}
        )
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(
                f"No samples found for split='{split}' in {self.manifest_path}"
            )

        if self.class_to_idx:
            self._validate_samples_against_mapping()
        else:
            self.class_to_idx = self._mapping_from_samples()

        self.idx_to_class = {label: name for name, label in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]
        self.visible_classes = sorted({sample.class_name for sample in self.samples})
        LOGGER.info(
            "Loaded AwA2 split=%s with %d samples, %d visible classes and %d mapped classes",
            split,
            len(self.samples),
            len(self.visible_classes),
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

    def _mapping_from_samples(self) -> dict[str, int]:
        class_to_idx: dict[str, int] = {}
        for sample in self.samples:
            existing = class_to_idx.get(sample.class_name)
            if existing is not None and existing != sample.label:
                raise ValueError(
                    "Inconsistent labels for class "
                    f"{sample.class_name}: {existing} and {sample.label}"
                )
            class_to_idx[sample.class_name] = sample.label
        return dict(sorted(class_to_idx.items(), key=lambda item: item[1]))

    def _validate_samples_against_mapping(self) -> None:
        for sample in self.samples:
            expected_label = self.class_to_idx.get(sample.class_name)
            if expected_label is None:
                raise ValueError(
                    f"Class {sample.class_name} is missing from {self.class_map_path}"
                )
            if sample.label != expected_label:
                raise ValueError(
                    "Manifest label does not match class mapping for "
                    f"{sample.class_name}: manifest={sample.label}, "
                    f"mapping={expected_label}"
                )

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
    class_map_path: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from the AwA2 manifest."""
    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = AwA2Dataset(
            manifest_path=manifest_path,
            split=split,
            transform=build_resnet_transforms(train=split == "train"),
            class_map_path=class_map_path,
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
