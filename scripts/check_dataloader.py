"""Smoke test for AwA2 Dataset/DataLoader and normalization ranges."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    AwA2Dataset,
    build_dataloaders,
    build_debug_transform,
    denormalize_batch,
    infer_class_map_path,
    load_class_mapping,
)
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("check_dataloader")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AwA2 DataLoader setup.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2" / "awa2_manifest.csv",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional class_to_idx CSV. If omitted, the script infers it.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def print_manifest_summary(manifest_path: Path) -> None:
    split_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    examples: list[dict[str, str]] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_counter[row["split"]] += 1
            class_counter[row["class_name"]] += 1
            if len(examples) < 5:
                examples.append(row)

    LOGGER.info("Manifest split counts: %s", dict(split_counter))
    LOGGER.info("Number of classes in manifest: %d", len(class_counter))
    LOGGER.info("First manifest rows:")
    for row in examples:
        LOGGER.info(
            "  split=%s label=%s class=%s path=%s",
            row["split"],
            row["label"],
            row["class_name"],
            row["filepath"],
        )


def print_class_mapping_examples(class_map_path: Path | None) -> None:
    if class_map_path is None:
        LOGGER.warning("No class mapping CSV found next to the manifest")
        return

    class_to_idx = load_class_mapping(class_map_path)
    examples = list(class_to_idx.items())[:10]
    LOGGER.info("Class mapping file: %s", class_map_path)
    LOGGER.info("Number of mapped classes: %d", len(class_to_idx))
    LOGGER.info("First class mapping rows:")
    for class_name, label in examples:
        LOGGER.info("  label=%d class=%s", label, class_name)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    class_map = (
        args.class_map.expanduser().resolve()
        if args.class_map is not None
        else infer_class_map_path(manifest)
    )

    print_manifest_summary(manifest)
    print_class_mapping_examples(class_map)

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        class_map_path=class_map,
    )

    images, labels, class_names, paths = next(iter(loaders["train"]))
    LOGGER.info("Batch tensor shape after ImageNet normalization: %s", tuple(images.shape))
    LOGGER.info("Labels shape: %s", tuple(labels.shape))
    LOGGER.info("First class names in batch: %s", list(class_names[:5]))
    LOGGER.info("First image path: %s", paths[0])
    LOGGER.info(
        "Normalized batch stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
        images.min().item(),
        images.max().item(),
        images.mean().item(),
        images.std().item(),
    )

    denorm = denormalize_batch(images).clamp(0.0, 1.0)
    LOGGER.info(
        "Denormalized batch stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
        denorm.min().item(),
        denorm.max().item(),
        denorm.mean().item(),
        denorm.std().item(),
    )

    debug_dataset = AwA2Dataset(
        manifest_path=manifest,
        split="train",
        transform=build_debug_transform(),
        class_map_path=class_map,
    )
    raw_tensor, raw_label, raw_class, _ = debug_dataset[0]
    LOGGER.info(
        "Pre-normalization sample: shape=%s label=%d class=%s min=%.4f max=%.4f",
        tuple(raw_tensor.shape),
        raw_label,
        raw_class,
        raw_tensor.min().item(),
        raw_tensor.max().item(),
    )


if __name__ == "__main__":
    main()
