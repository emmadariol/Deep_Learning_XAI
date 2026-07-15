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
        class_map_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_dir = self.manifest_path.parent
        self.class_map_path = (
            Path(class_map_path).expanduser().resolve()
            if class_map_path is not None
            else None
        )
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' in {self.manifest_path}")

        self.visible_classes = sorted({sample.class_name for sample in self.samples})
        if self.class_map_path is not None:
            self.idx_to_class = load_class_mapping(self.class_map_path)
            self.classes = [self.idx_to_class[index] for index in sorted(self.idx_to_class)]
        else:
            self.idx_to_class = {
                index: class_name for index, class_name in enumerate(self.visible_classes)
            }
            self.classes = list(self.visible_classes)

        LOGGER.info(
            "Loaded manifest split=%s with %d samples, %d visible classes, %d mapped classes from %s",
            split,
            len(self.samples),
            len(self.visible_classes),
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


def infer_class_map_path(manifest_path: str | Path) -> Path:
    """Infer the class-to-index CSV path associated with a manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest.parent
    candidates = [
        manifest_dir / "class_to_idx.csv",
        manifest_dir / "class_to_idx_debug.csv",
        manifest_dir / "class_to_idx_subset.csv",
    ]

    if "debug" in manifest.stem:
        candidates.insert(0, manifest_dir / "class_to_idx_debug.csv")
    if "subset" in manifest.stem:
        candidates.insert(0, manifest_dir / "class_to_idx_subset.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not infer class map for {manifest}. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_class_mapping(class_map_path: str | Path) -> dict[int, str]:
    """Load a class mapping CSV as label -> class_name."""
    path = Path(class_map_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    mapping: dict[int, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"class_name", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Class map is missing columns: {sorted(missing)}")

        for row in reader:
            mapping[int(row["label"])] = row["class_name"]

    if not mapping:
        raise ValueError(f"No class mappings found in {path}")
    return dict(sorted(mapping.items()))


def infer_num_classes(manifest_path: str | Path, require_contiguous: bool = True) -> int:
    """Infer the number of classes from manifest labels."""
    labels: set[int] = set()
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "label" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest is missing column: label")
        for row in reader:
            labels.add(int(row["label"]))

    if not labels:
        raise ValueError(f"No labels found in manifest: {path}")
    expected = set(range(max(labels) + 1))
    if require_contiguous and labels != expected:
        raise ValueError(f"Labels are not contiguous from 0 to {max(labels)}.")
    return max(labels) + 1


def load_class_names(manifest_path: str | Path) -> dict[int, str]:
    """Load label -> class_name from an image manifest."""
    names: dict[int, str] = {}
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "class_name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            names[int(row["label"])] = row["class_name"]

    if not names:
        raise ValueError(f"No class names found in manifest: {path}")
    return dict(sorted(names.items()))


def load_idx_to_class(manifest_path: str | Path) -> dict[int, str]:
    """Compatibility alias for label -> class_name manifest loading."""
    return load_class_names(manifest_path)


def names_from_labels(labels: torch.Tensor, idx_to_class: dict[int, str]) -> list[str]:
    """Convert a tensor of integer labels to class names."""
    return [idx_to_class[int(label.item())] for label in labels.detach().cpu()]


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
    class_map_path: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from an image-classification manifest."""
    dataloaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = ImageManifestDataset(
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
