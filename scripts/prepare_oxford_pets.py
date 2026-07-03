"""Prepare Oxford-IIIT Pet images and trimaps into CSV manifests."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import resolve_path, set_seed, setup_logging

LOGGER = logging.getLogger("prepare_oxford_pets")
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Oxford-IIIT Pet manifests from images/ and annotations/."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "OxfordPets",
        help="Dataset root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "OxfordPets",
        help="Where oxford_pets_manifest.csv and class_to_idx.csv will be written.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official images.tar.gz and annotations.tar.gz if missing.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Use only the first N sorted breeds. Useful for debug manifests.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Use at most N images per breed before writing the manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="oxford_pets_manifest.csv",
        help="Output manifest filename.",
    )
    parser.add_argument(
        "--class-map-name",
        type=str,
        default="class_to_idx.csv",
        help="Output class mapping filename.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s", url)
    LOGGER.info("Destination: %s", destination)
    urllib.request.urlretrieve(url, destination)


def extract_tar(archive_path: Path, data_root: Path) -> None:
    LOGGER.info("Extracting %s into %s", archive_path, data_root)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(data_root)


def ensure_dataset(data_root: Path, download: bool) -> tuple[Path, Path, Path]:
    images_dir = data_root / "images"
    annotations_dir = data_root / "annotations"
    trimaps_dir = annotations_dir / "trimaps"

    if images_dir.is_dir() and trimaps_dir.is_dir():
        return images_dir, annotations_dir, trimaps_dir

    if not download:
        raise FileNotFoundError(
            f"Could not find Oxford-IIIT Pet images/ and annotations/ under {data_root}. "
            "Place the official extracted folders there or rerun with --download."
        )

    images_archive = data_root / "images.tar.gz"
    annotations_archive = data_root / "annotations.tar.gz"
    if not images_archive.exists():
        download_file(IMAGES_URL, images_archive)
    if not annotations_archive.exists():
        download_file(ANNOTATIONS_URL, annotations_archive)

    extract_tar(images_archive, data_root)
    extract_tar(annotations_archive, data_root)

    if not images_dir.is_dir() or not trimaps_dir.is_dir():
        raise FileNotFoundError("Download/extraction finished, but images/ or trimaps/ is missing.")

    return images_dir, annotations_dir, trimaps_dir


def parse_split_file(split_path: Path, split_name: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            image_id, class_id, species_id, breed_id = stripped.split()
            class_name = image_id.rsplit("_", 1)[0]
            rows.append(
                {
                    "image_id": image_id,
                    "official_class_id": int(class_id),
                    "official_species_id": int(species_id),
                    "official_breed_id": int(breed_id),
                    "class_name": class_name,
                    "species": "cat" if int(species_id) == 1 else "dog",
                    "official_split": split_name,
                }
            )
    return rows


def apply_subset(
    rows: list[dict[str, str | int]],
    max_classes: int | None,
    max_images_per_class: int | None,
    seed: int,
) -> list[dict[str, str | int]]:
    selected_classes = sorted({str(row["class_name"]) for row in rows})
    if max_classes is not None:
        if max_classes <= 0:
            raise ValueError("--max-classes must be positive")
        selected_classes = selected_classes[:max_classes]

    by_class: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in rows:
        if str(row["class_name"]) in selected_classes:
            by_class[str(row["class_name"])].append(row)

    rng = random.Random(seed)
    selected_rows: list[dict[str, str | int]] = []
    for class_name in selected_classes:
        class_rows = list(by_class[class_name])
        class_rows.sort(key=lambda item: str(item["image_id"]))
        if max_images_per_class is not None:
            if max_images_per_class <= 0:
                raise ValueError("--max-images-per-class must be positive")
            rng.shuffle(class_rows)
            class_rows = sorted(class_rows[:max_images_per_class], key=lambda item: str(item["image_id"]))
        selected_rows.extend(class_rows)

    if max_classes is not None or max_images_per_class is not None:
        LOGGER.info(
            "Subset active: %d classes, %d total images, max_images_per_class=%s",
            len(selected_classes),
            len(selected_rows),
            max_images_per_class,
        )

    return selected_rows


def assign_splits(
    rows: list[dict[str, str | int]],
    val_ratio: float,
    seed: int,
) -> list[dict[str, str | int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    trainval_by_class: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    final_rows: list[dict[str, str | int]] = []

    for row in rows:
        if row["official_split"] == "test":
            final_rows.append({**row, "split": "test"})
        else:
            trainval_by_class[str(row["class_name"])].append(row)

    rng = random.Random(seed)
    for class_name, class_rows in trainval_by_class.items():
        shuffled = list(class_rows)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        val_ids = {str(row["image_id"]) for row in shuffled[:val_count]}
        for row in class_rows:
            split = "val" if str(row["image_id"]) in val_ids else "train"
            final_rows.append({**row, "split": split})

    return sorted(final_rows, key=lambda item: (str(item["split"]), str(item["class_name"]), str(item["image_id"])))


def write_manifests(
    rows: list[dict[str, str | int]],
    images_dir: Path,
    trimaps_dir: Path,
    manifest_dir: Path,
    manifest_name: str,
    class_map_name: str,
) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / manifest_name
    class_map_path = manifest_dir / class_map_name
    class_names = sorted({str(row["class_name"]) for row in rows})
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    split_counts = {"train": 0, "val": 0, "test": 0}
    missing_assets: list[str] = []

    with class_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "label"])
        writer.writeheader()
        for class_name, label in class_to_idx.items():
            writer.writerow({"class_name": class_name, "label": label})

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "filepath",
            "mask_path",
            "label",
            "class_name",
            "species",
            "split",
            "image_id",
            "official_class_id",
            "official_species_id",
            "official_breed_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            image_id = str(row["image_id"])
            image_path = images_dir / f"{image_id}.jpg"
            mask_path = trimaps_dir / f"{image_id}.png"
            if not image_path.exists() or not mask_path.exists():
                missing_assets.append(image_id)
                continue

            split = str(row["split"])
            split_counts[split] += 1
            writer.writerow(
                {
                    "filepath": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "label": class_to_idx[str(row["class_name"])],
                    "class_name": row["class_name"],
                    "species": row["species"],
                    "split": split,
                    "image_id": image_id,
                    "official_class_id": row["official_class_id"],
                    "official_species_id": row["official_species_id"],
                    "official_breed_id": row["official_breed_id"],
                }
            )

    if missing_assets:
        LOGGER.warning("Skipped %d rows with missing image/trimap assets", len(missing_assets))
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
    images_dir, annotations_dir, trimaps_dir = ensure_dataset(data_root, args.download)

    trainval_rows = parse_split_file(annotations_dir / "trainval.txt", "trainval")
    test_rows = parse_split_file(annotations_dir / "test.txt", "test")
    rows = trainval_rows + test_rows
    rows = apply_subset(
        rows=rows,
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
    )
    rows = assign_splits(rows=rows, val_ratio=args.val_ratio, seed=args.seed)

    LOGGER.info(
        "Collected %d Oxford-IIIT Pet rows across %d classes",
        len(rows),
        len({row["class_name"] for row in rows}),
    )
    write_manifests(
        rows=rows,
        images_dir=images_dir,
        trimaps_dir=trimaps_dir,
        manifest_dir=manifest_dir,
        manifest_name=args.manifest_name,
        class_map_name=args.class_map_name,
    )


if __name__ == "__main__":
    main()

