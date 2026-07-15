"""Prepare full AwA2 manifests or portable image subsets from one CLI.

``--mode manifest`` writes a manifest for an existing AwA2 image tree and can
download the official archive. ``--mode subset`` copies, links, or resizes a
deterministic subset into a portable folder with its own manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import resolve_path, set_seed, setup_logging

LOGGER = logging.getLogger("prepare_awa2")
DEFAULT_AWA2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BACKGROUND20_CLASSES = [
    "antelope",
    "blue+whale",
    "bobcat",
    "buffalo",
    "dolphin",
    "elephant",
    "giant+panda",
    "giraffe",
    "grizzly+bear",
    "hippopotamus",
    "humpback+whale",
    "leopard",
    "lion",
    "polar+bear",
    "seal",
    "sheep",
    "tiger",
    "walrus",
    "wolf",
    "zebra",
]

DEBUG10_CLASSES = [
    "antelope",
    "dolphin",
    "elephant",
    "giraffe",
    "grizzly+bear",
    "polar+bear",
    "seal",
    "tiger",
    "wolf",
    "zebra",
]


@dataclass(frozen=True)
class ManifestSample:
    filepath: str
    label: int
    class_name: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create AwA2 manifests or portable image subsets.",
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "subset"],
        default="manifest",
        help="Use manifest for the original image tree or subset for a copied portable dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")

    manifest = parser.add_argument_group("manifest mode")
    manifest.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2",
        help="AwA2 root containing JPEGImages/, or where --download is extracted.",
    )
    manifest.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2",
        help="Directory where manifest and class mapping are written.",
    )
    manifest.add_argument(
        "--download",
        action="store_true",
        help="Download AwA2-data.zip if JPEGImages/ is missing (manifest mode only).",
    )
    manifest.add_argument("--url", type=str, default=DEFAULT_AWA2_URL)
    manifest.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Use the first N sorted classes in manifest mode.",
    )
    manifest.add_argument(
        "--manifest-name",
        type=str,
        default="awa2_manifest.csv",
    )
    manifest.add_argument(
        "--class-map-name",
        type=str,
        default="class_to_idx.csv",
    )

    subset = parser.add_argument_group("subset mode")
    subset.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Original AwA2 root, or its JPEGImages directory.",
    )
    subset.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination folder for the portable subset.",
    )
    subset.add_argument(
        "--preset",
        choices=["background20", "debug10", "none"],
        default="background20",
        help="Subset class preset. Ignored when --classes or --classes-file is provided.",
    )
    subset.add_argument("--classes", type=str, default=None)
    subset.add_argument("--classes-file", type=Path, default=None)
    subset.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Classes to sample when --preset none has no explicit class list.",
    )
    subset.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
    )
    subset.add_argument("--allow-existing", action="store_true")
    subset.add_argument("--make-zip", action="store_true")
    subset.add_argument("--resize-size", type=int, default=None)
    subset.add_argument(
        "--resize-method",
        choices=["pad", "crop", "stretch"],
        default="pad",
    )
    subset.add_argument("--jpeg-quality", type=int, default=92)
    subset.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Limit each class; default is unlimited in manifest mode and 200 in subset mode.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("+", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading AwA2 archive from %s", url)
    urllib.request.urlretrieve(url, destination)


def extract_archive(archive_path: Path, data_root: Path) -> None:
    LOGGER.info("Extracting %s into %s", archive_path, data_root)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(data_root)


def find_jpeg_images_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    if root.name == "JPEGImages" and root.is_dir():
        return root

    direct = root / "JPEGImages"
    if direct.is_dir():
        return direct

    candidates = sorted(path for path in root.rglob("JPEGImages") if path.is_dir())
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find JPEGImages/ under {root}.")


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
    if not class_to_images:
        raise ValueError(f"No image classes found under {jpeg_dir}.")
    if len(class_to_images) != 50:
        LOGGER.warning(
            "Expected 50 AwA2 classes, found %d under %s",
            len(class_to_images),
            jpeg_dir,
        )
    return class_to_images


def select_manifest_images(
    class_to_images: dict[str, list[Path]],
    max_classes: int | None,
    max_images_per_class: int | None,
    seed: int,
) -> dict[str, list[Path]]:
    class_names = sorted(class_to_images)
    if max_classes is not None:
        if max_classes <= 0:
            raise ValueError("--max-classes must be positive.")
        class_names = class_names[:max_classes]

    rng = random.Random(seed)
    selected: dict[str, list[Path]] = {}
    for class_name in class_names:
        images = list(class_to_images[class_name])
        if max_images_per_class is not None:
            if max_images_per_class <= 0:
                raise ValueError("--max-images-per-class must be positive.")
            rng.shuffle(images)
            images = sorted(images[:max_images_per_class])
        selected[class_name] = images
    return selected


def read_requested_classes(args: argparse.Namespace) -> list[str] | None:
    if args.classes_file is not None:
        with args.classes_file.expanduser().open("r", encoding="utf-8") as handle:
            return [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    if args.classes is not None:
        return [item.strip() for item in args.classes.split(",") if item.strip()]
    if args.preset == "background20":
        return BACKGROUND20_CLASSES
    if args.preset == "debug10":
        return DEBUG10_CLASSES
    return None


def select_subset_classes(
    args: argparse.Namespace,
    class_to_images: dict[str, list[Path]],
) -> list[str]:
    available = sorted(class_to_images)
    requested = read_requested_classes(args)
    if requested is None:
        if args.num_classes <= 0 or args.num_classes > len(available):
            raise ValueError(f"--num-classes must be between 1 and {len(available)}.")
        return sorted(random.Random(args.seed).sample(available, args.num_classes))

    lookup = {normalize_name(name): name for name in available}
    selected: list[str] = []
    missing: list[str] = []
    for name in requested:
        match = lookup.get(normalize_name(name))
        if match is None:
            missing.append(name)
        else:
            selected.append(match)
    if missing:
        raise ValueError(f"Requested classes not found in JPEGImages: {missing}")
    selected = sorted(set(selected))
    if not selected:
        raise ValueError("No classes selected.")
    return selected


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
    ensure_eval_splits: bool = False,
) -> dict[str, list[Path]]:
    """Split one class deterministically, optionally keeping val/test non-empty."""
    shuffled = list(images)
    rng.shuffle(shuffled)
    total = len(shuffled)
    if ensure_eval_splits and total == 1:
        return {"train": shuffled, "val": [], "test": []}
    if ensure_eval_splits and total == 2:
        return {"train": shuffled[:1], "val": [], "test": shuffled[1:]}
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if ensure_eval_splits and total >= 3:
        val_count = max(1, round(total * val_ratio))
        test_count = max(1, round(total * test_ratio))
        train_count = total - val_count - test_count
        while train_count < 1:
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            train_count = total - val_count - test_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def write_class_map(class_to_idx: dict[str, int], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "label"])
        writer.writeheader()
        for class_name, label in class_to_idx.items():
            writer.writerow({"class_name": class_name, "label": label})
    return path


def write_manifest(samples: list[ManifestSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filepath", "label", "class_name", "split"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return path


def run_manifest_mode(args: argparse.Namespace) -> None:
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

    class_to_images = select_manifest_images(
        collect_class_images(jpeg_dir),
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
    )
    class_to_idx = {name: index for index, name in enumerate(sorted(class_to_images))}
    rng = random.Random(args.seed)
    samples: list[ManifestSample] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for class_name, images in class_to_images.items():
        for split, split_images_for_class in split_images(
            images,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            rng,
        ).items():
            split_counts[split] += len(split_images_for_class)
            samples.extend(
                ManifestSample(
                    filepath=str(image_path.resolve()),
                    label=class_to_idx[class_name],
                    class_name=class_name,
                    split=split,
                )
                for image_path in split_images_for_class
            )

    manifest_path = write_manifest(samples, manifest_dir / args.manifest_name)
    class_map_path = write_class_map(class_to_idx, manifest_dir / args.class_map_name)
    LOGGER.info("Using JPEGImages directory: %s", jpeg_dir)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote class mapping: %s", class_map_path)
    LOGGER.info("Split counts: %s", split_counts)


def prepare_output_root(output_root: Path, allow_existing: bool) -> Path:
    output_root = resolve_path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_root}. "
            "Use a new folder or pass --allow-existing."
        )
    (output_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    return output_root


def resize_image(
    source: Path,
    destination: Path,
    resize_size: int,
    resize_method: str,
    jpeg_quality: int,
) -> None:
    from PIL import Image, ImageOps

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        target_size = (resize_size, resize_size)
        if resize_method == "pad":
            resized = ImageOps.contain(image, target_size, method=resampling)
            output = Image.new("RGB", target_size, color=(124, 116, 104))
            output.paste(
                resized,
                (
                    (resize_size - resized.width) // 2,
                    (resize_size - resized.height) // 2,
                ),
            )
        elif resize_method == "crop":
            output = ImageOps.fit(image, target_size, method=resampling, centering=(0.5, 0.5))
        elif resize_method == "stretch":
            output = image.resize(target_size, resample=resampling)
        else:
            raise ValueError(f"Unsupported resize method: {resize_method}")
        save_kwargs: dict[str, int | bool] = {}
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": jpeg_quality, "optimize": True}
        output.save(destination, **save_kwargs)


def copy_image(
    source: Path,
    destination: Path,
    copy_mode: str,
    resize_size: int | None,
    resize_method: str,
    jpeg_quality: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if resize_size is not None:
        resize_image(source, destination, resize_size, resize_method, jpeg_quality)
    elif copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        destination.hardlink_to(source)
    elif copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def write_subset_summary(
    args: argparse.Namespace,
    output_root: Path,
    jpeg_dir: Path,
    selected_classes: list[str],
    samples: list[ManifestSample],
    max_images_per_class: int,
) -> Path:
    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in ("train", "val", "test")
    }
    class_split_counts = {
        class_name: {
            split: sum(
                sample.class_name == class_name and sample.split == split
                for sample in samples
            )
            for split in ("train", "val", "test")
        }
        for class_name in selected_classes
    }
    path = output_root / "subset_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jpeg_dir": str(jpeg_dir),
                "output_root": str(output_root),
                "seed": args.seed,
                "copy_mode": args.copy_mode,
                "resize_size": args.resize_size,
                "resize_method": args.resize_method if args.resize_size is not None else None,
                "jpeg_quality": args.jpeg_quality if args.resize_size is not None else None,
                "max_images_per_class": max_images_per_class,
                "ratios": {
                    "train": args.train_ratio,
                    "val": args.val_ratio,
                    "test": args.test_ratio,
                },
                "num_classes": len(selected_classes),
                "num_images": len(samples),
                "split_counts": split_counts,
                "classes": selected_classes,
                "class_split_counts": class_split_counts,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return path


def write_subset_readme(output_root: Path) -> Path:
    path = output_root / "README_subset.md"
    path.write_text(
        """# AwA2 Subset

This portable subset was generated with `scripts/data/prepare_awa2.py --mode subset`.

- `JPEGImages/`: subset images organized by class.
- `awa2_manifest_subset.csv`: relative image paths, labels, class names and split.
- `class_to_idx_subset.csv`: stable class-to-label mapping.
- `subset_summary.json`: generation configuration and counts.
""",
        encoding="utf-8",
    )
    return path


def create_zip(output_root: Path) -> Path:
    zip_path = output_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_root.parent))
    return zip_path


def run_subset_mode(args: argparse.Namespace) -> None:
    if args.source_root is None or args.output_root is None:
        raise ValueError("--source-root and --output-root are required in subset mode.")
    max_images_per_class = 200 if args.max_images_per_class is None else args.max_images_per_class
    if max_images_per_class <= 0:
        raise ValueError("--max-images-per-class must be positive.")
    if args.resize_size is not None and args.resize_size <= 0:
        raise ValueError("--resize-size must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if args.resize_size is not None and args.copy_mode != "copy":
        raise ValueError("--resize-size requires --copy-mode copy.")

    jpeg_dir = find_jpeg_images_dir(args.source_root)
    class_to_images = collect_class_images(jpeg_dir)
    selected_classes = select_subset_classes(args, class_to_images)
    LOGGER.info("Using JPEGImages directory: %s", jpeg_dir)
    for class_name in selected_classes:
        LOGGER.info(
            "Selected %s: %d/%d images",
            class_name,
            min(len(class_to_images[class_name]), max_images_per_class),
            len(class_to_images[class_name]),
        )
    if args.dry_run:
        LOGGER.info("Dry run complete. No files copied.")
        return

    output_root = prepare_output_root(args.output_root, args.allow_existing)
    class_to_idx = {class_name: index for index, class_name in enumerate(selected_classes)}
    rng = random.Random(args.seed)
    samples: list[ManifestSample] = []
    for class_name in selected_classes:
        class_images = list(class_to_images[class_name])
        rng.shuffle(class_images)
        selected_images = sorted(class_images[:max_images_per_class])
        split_to_images = split_images(
            selected_images,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            rng,
            ensure_eval_splits=True,
        )
        for split, split_images_for_class in split_to_images.items():
            for source_path in split_images_for_class:
                destination = output_root / "JPEGImages" / class_name / source_path.name
                copy_image(
                    source_path,
                    destination,
                    args.copy_mode,
                    args.resize_size,
                    args.resize_method,
                    args.jpeg_quality,
                )
                samples.append(
                    ManifestSample(
                        filepath=destination.relative_to(output_root).as_posix(),
                        label=class_to_idx[class_name],
                        class_name=class_name,
                        split=split,
                    )
                )

    manifest_path = write_manifest(samples, output_root / "awa2_manifest_subset.csv")
    class_map_path = write_class_map(class_to_idx, output_root / "class_to_idx_subset.csv")
    summary_path = write_subset_summary(
        args,
        output_root,
        jpeg_dir,
        selected_classes,
        samples,
        max_images_per_class,
    )
    readme_path = write_subset_readme(output_root)
    LOGGER.info("Wrote subset: %s", output_root)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote class mapping: %s", class_map_path)
    LOGGER.info("Wrote summary: %s", summary_path)
    LOGGER.info("Wrote README: %s", readme_path)
    if args.make_zip:
        LOGGER.info("Wrote zip archive: %s", create_zip(output_root))


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    setup_logging(args.log_level)
    set_seed(args.seed)
    if args.mode == "manifest":
        run_manifest_mode(args)
    else:
        run_subset_mode(args)


if __name__ == "__main__":
    main()
