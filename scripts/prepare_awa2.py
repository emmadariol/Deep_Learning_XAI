"""Prepare AwA2 JPEGImages into reproducible train/val/test CSV manifests."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import resolve_path, set_seed, setup_logging

LOGGER = logging.getLogger("prepare_awa2")
DEFAULT_AWA2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create AwA2 manifests from JPEGImages, optionally downloading AwA2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2",
        help="AwA2 root folder containing JPEGImages/ or where the archive is extracted.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2",
        help="Where awa2_manifest.csv and class_to_idx.csv will be written.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download AwA2-data.zip if JPEGImages/ is missing.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_AWA2_URL,
        help="AwA2 archive URL. Override this if the official mirror changes.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading AwA2 archive from %s", url)
    LOGGER.info("Destination: %s", destination)
    urllib.request.urlretrieve(url, destination)


def extract_archive(archive_path: Path, data_root: Path) -> None:
    LOGGER.info("Extracting %s into %s", archive_path, data_root)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(data_root)


def find_jpeg_images_dir(data_root: Path) -> Path:
    direct = data_root / "JPEGImages"
    if direct.is_dir():
        return direct

    candidates = [path for path in data_root.rglob("JPEGImages") if path.is_dir()]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find JPEGImages/ under {data_root}. "
        "Place AwA2 JPEGImages there or rerun with --download."
    )


def collect_class_images(jpeg_dir: Path) -> dict[str, list[Path]]:
    class_to_images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in jpeg_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_to_images[class_dir.name] = images

    if len(class_to_images) != 50:
        LOGGER.warning(
            "Expected 50 AwA2 classes, found %d under %s",
            len(class_to_images),
            jpeg_dir,
        )

    return class_to_images


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    shuffled = list(images)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_manifests(
    class_to_images: dict[str, list[Path]],
    manifest_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "awa2_manifest.csv"
    class_map_path = manifest_dir / "class_to_idx.csv"
    rng = random.Random(seed)
    class_names = sorted(class_to_images)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    with class_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "label"])
        writer.writeheader()
        for class_name, label in class_to_idx.items():
            writer.writerow({"class_name": class_name, "label": label})

    split_counts = {"train": 0, "val": 0, "test": 0}
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filepath", "label", "class_name", "split"]
        )
        writer.writeheader()
        for class_name in class_names:
            split_to_images = split_images(
                class_to_images[class_name],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                rng=rng,
            )
            for split, images in split_to_images.items():
                split_counts[split] += len(images)
                for image_path in images:
                    writer.writerow(
                        {
                            "filepath": str(image_path.resolve()),
                            "label": class_to_idx[class_name],
                            "class_name": class_name,
                            "split": split,
                        }
                    )

    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote class mapping: %s", class_map_path)
    LOGGER.info("Split counts: %s", split_counts)
    return manifest_path, class_map_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    manifest_dir = resolve_path(args.manifest_dir)

    try:
        jpeg_dir = find_jpeg_images_dir(data_root)
    except FileNotFoundError:
        if not args.download:
            raise
        archive_path = data_root / "AwA2-data.zip"
        download_archive(args.url, archive_path)
        extract_archive(archive_path, data_root)
        jpeg_dir = find_jpeg_images_dir(data_root)

    LOGGER.info("Using JPEGImages directory: %s", jpeg_dir)
    class_to_images = collect_class_images(jpeg_dir)
    total_images = sum(len(images) for images in class_to_images.values())
    LOGGER.info("Collected %d images across %d classes", total_images, len(class_to_images))

    write_manifests(
        class_to_images=class_to_images,
        manifest_dir=manifest_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

