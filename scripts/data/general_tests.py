"""Minimal smoke tests for a manifest-based image-classification pipeline."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    build_dataloaders,
    denormalize_batch,
    infer_class_map_path,
    infer_num_classes,
    load_class_mapping,
)
from src.utils import setup_logging

LOGGER = logging.getLogger("general_tests")
REQUIRED_COLUMNS = {"filepath", "label", "class_name", "split"}
REQUIRED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal general checks for an image-classification manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with filepath, label, class_name, and split columns.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help="Optional class_name,label CSV; inferred beside the manifest when available.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def validate_manifest(manifest_path: Path) -> dict[int, str]:
    """Check schema, labels, split coverage, and referenced image files."""
    split_counts: Counter[str] = Counter()
    label_to_class: dict[int, str] = {}
    class_to_label: dict[str, int] = {}

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            if not all(row[column] for column in REQUIRED_COLUMNS):
                raise ValueError(f"Manifest row {row_number} has an empty required field.")
            try:
                label = int(row["label"])
            except ValueError as error:
                raise ValueError(
                    f"Manifest row {row_number} has a non-integer label: {row['label']!r}"
                ) from error
            if label < 0:
                raise ValueError(f"Manifest row {row_number} has a negative label: {label}")

            class_name = row["class_name"]
            if label_to_class.setdefault(label, class_name) != class_name:
                raise ValueError(f"Label {label} maps to multiple class names.")
            if class_to_label.setdefault(class_name, label) != label:
                raise ValueError(f"Class {class_name!r} maps to multiple labels.")

            image_path = Path(row["filepath"]).expanduser()
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Manifest row {row_number} references a missing image: {image_path}"
                )
            split_counts[row["split"]] += 1

    if not label_to_class:
        raise ValueError(f"Manifest has no samples: {manifest_path}")
    missing_splits = set(REQUIRED_SPLITS).difference(split_counts)
    if missing_splits:
        raise ValueError(f"Manifest is missing required splits: {sorted(missing_splits)}")
    if set(label_to_class) != set(range(max(label_to_class) + 1)):
        raise ValueError("Manifest labels must be contiguous and start from 0.")
    if infer_num_classes(manifest_path) != len(label_to_class):
        raise AssertionError("Class-count inference disagrees with manifest labels.")

    LOGGER.info("PASS manifest: classes=%d splits=%s", len(label_to_class), dict(split_counts))
    return label_to_class


def resolve_class_map(manifest_path: Path, class_map: Path | None) -> Path | None:
    if class_map is not None:
        return class_map.expanduser().resolve()
    try:
        return infer_class_map_path(manifest_path)
    except FileNotFoundError:
        LOGGER.info("No adjacent class mapping found; checking manifest labels only.")
        return None


def validate_batches(
    manifest_path: Path,
    class_map_path: Path | None,
    label_to_class: dict[int, str],
    batch_size: int,
    num_workers: int,
) -> None:
    """Load one batch per split and validate shapes, metadata, and normalization."""
    loaders = build_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        class_map_path=class_map_path,
    )
    for split in REQUIRED_SPLITS:
        images, labels, class_names, paths = next(iter(loaders[split]))
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise AssertionError(f"{split} batch has unexpected image shape: {tuple(images.shape)}")
        if labels.ndim != 1 or labels.size(0) != images.size(0):
            raise AssertionError(f"{split} batch has incompatible labels: {tuple(labels.shape)}")
        if not torch.isfinite(images).all():
            raise AssertionError(f"{split} batch contains non-finite image values.")
        if len(class_names) != images.size(0) or len(paths) != images.size(0):
            raise AssertionError(f"{split} batch metadata does not match its images.")

        for label, class_name, path in zip(labels.tolist(), class_names, paths, strict=True):
            if label_to_class.get(label) != class_name:
                raise AssertionError(f"{split} batch has an inconsistent label/class pair.")
            if not Path(path).is_file():
                raise FileNotFoundError(f"{split} batch returned a missing image: {path}")

        restored = denormalize_batch(images)
        if not torch.isfinite(restored).all() or not (
            restored.min().item() >= -1e-5 and restored.max().item() <= 1.0 + 1e-5
        ):
            raise AssertionError(f"{split} normalization round-trip is invalid.")
        LOGGER.info("PASS %s batch: shape=%s", split, tuple(images.shape))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    manifest_path = args.manifest.expanduser().resolve()
    label_to_class = validate_manifest(manifest_path)
    class_map_path = resolve_class_map(manifest_path, args.class_map)
    if class_map_path is not None:
        if load_class_mapping(class_map_path) != label_to_class:
            raise ValueError(f"Class mapping does not match the manifest: {class_map_path}")
        LOGGER.info("PASS class mapping: %s", class_map_path)

    validate_batches(
        manifest_path,
        class_map_path,
        label_to_class,
        args.batch_size,
        args.num_workers,
    )
    LOGGER.info("All general tests passed.")


if __name__ == "__main__":
    main()
