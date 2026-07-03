"""Create portable, reproducible image subsets from the original AwA2 dataset.

The script expects an AwA2 folder containing ``JPEGImages/<class_name>/*.jpg``.
It copies a deterministic subset of images, writes a portable manifest with
relative paths, and optionally creates a zip archive that can be shared.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


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
class CopiedSample:
    filepath: str
    label: int
    class_name: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight AwA2 image subset with JPEGImages/, "
            "awa2_manifest_subset.csv and class_to_idx_subset.csv."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Original AwA2 root, or the JPEGImages directory itself.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination folder for the subset to share.",
    )
    parser.add_argument(
        "--preset",
        choices=["background20", "debug10", "none"],
        default="background20",
        help=(
            "Class preset. Use 'none' with --classes/--classes-file, "
            "or with --num-classes for deterministic random selection."
        ),
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names. Overrides --preset.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="Optional text file with one class name per line. Overrides --preset.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of classes to sample when --preset none and no class list is given.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=200,
        help="Maximum images copied per selected class.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="Use 'copy' for a portable subset. Links are useful only locally.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow writing into an existing non-empty output folder.",
    )
    parser.add_argument(
        "--make-zip",
        action="store_true",
        help="Create output-root.zip after the subset is generated.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help=(
            "Optionally save each image as a square SIZE x SIZE image. "
            "The default is no resizing."
        ),
    )
    parser.add_argument(
        "--resize-method",
        choices=["pad", "crop", "stretch"],
        default="pad",
        help=(
            "Resize strategy when --resize-size is set. 'pad' preserves the "
            "whole image, 'crop' fills the square with center crop, and "
            "'stretch' distorts to the target size."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality used when resized images are saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected classes and counts without copying files.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("+", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_jpeg_images_dir(source_root: Path) -> Path:
    source_root = source_root.expanduser().resolve()
    if source_root.name == "JPEGImages" and source_root.is_dir():
        return source_root

    direct = source_root / "JPEGImages"
    if direct.is_dir():
        return direct

    candidates = sorted(path for path in source_root.rglob("JPEGImages") if path.is_dir())
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find JPEGImages/ under {source_root}. "
        "Expected AwA2/JPEGImages/<class_name>/*.jpg."
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
    if not class_to_images:
        raise ValueError(f"No image classes found under {jpeg_dir}")
    return class_to_images


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


def match_requested_classes(
    requested_classes: list[str],
    available_classes: list[str],
) -> list[str]:
    normalized_to_available = {
        normalize_name(class_name): class_name for class_name in available_classes
    }
    matched: list[str] = []
    missing: list[str] = []

    for requested in requested_classes:
        match = normalized_to_available.get(normalize_name(requested))
        if match is None:
            missing.append(requested)
        else:
            matched.append(match)

    if missing:
        raise ValueError(
            "Requested classes not found in AwA2 JPEGImages: "
            f"{missing}\nAvailable examples: {available_classes[:10]}"
        )

    return matched


def choose_classes(
    args: argparse.Namespace,
    class_to_images: dict[str, list[Path]],
) -> list[str]:
    available_classes = sorted(class_to_images)
    requested_classes = read_requested_classes(args)
    if requested_classes is not None:
        selected = match_requested_classes(requested_classes, available_classes)
    else:
        if args.num_classes <= 0:
            raise ValueError("--num-classes must be positive")
        if args.num_classes > len(available_classes):
            raise ValueError(
                f"--num-classes={args.num_classes} exceeds available classes "
                f"({len(available_classes)})"
            )
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(available_classes, args.num_classes))

    selected = sorted(set(selected))
    if not selected:
        raise ValueError("No classes selected.")
    return selected


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    shuffled = list(images)
    rng.shuffle(shuffled)
    total = len(shuffled)

    if total == 1:
        return {"train": shuffled, "val": [], "test": []}
    if total == 2:
        return {"train": shuffled[:1], "val": [], "test": shuffled[1:]}

    val_count = max(1, int(round(total * val_ratio)))
    test_count = max(1, int(round(total * test_ratio)))
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


def prepare_output_root(output_root: Path, allow_existing: bool) -> Path:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_root}\n"
            "Use a new folder or pass --allow-existing."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    return output_root


def resize_image(
    source: Path,
    destination: Path,
    resize_size: int,
    resize_method: str,
    jpeg_quality: int,
) -> None:
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for --resize-size. Install it with: pip install Pillow"
        ) from exc

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        target_size = (resize_size, resize_size)

        if resize_method == "pad":
            resized = ImageOps.contain(image, target_size, method=resampling)
            canvas = Image.new("RGB", target_size, color=(124, 116, 104))
            offset = (
                (resize_size - resized.width) // 2,
                (resize_size - resized.height) // 2,
            )
            canvas.paste(resized, offset)
            output = canvas
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
        resize_image(
            source=source,
            destination=destination,
            resize_size=resize_size,
            resize_method=resize_method,
            jpeg_quality=jpeg_quality,
        )
        return

    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        destination.hardlink_to(source)
    elif copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def write_class_map(class_to_idx: dict[str, int], output_root: Path) -> Path:
    path = output_root / "class_to_idx_subset.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "label"])
        writer.writeheader()
        for class_name, label in class_to_idx.items():
            writer.writerow({"class_name": class_name, "label": label})
    return path


def write_manifest(samples: list[CopiedSample], output_root: Path) -> Path:
    path = output_root / "awa2_manifest_subset.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filepath",
                "label",
                "class_name",
                "split",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return path


def write_summary(
    args: argparse.Namespace,
    output_root: Path,
    jpeg_dir: Path,
    selected_classes: list[str],
    samples: list[CopiedSample],
) -> Path:
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts: dict[str, dict[str, int]] = {
        class_name: {"train": 0, "val": 0, "test": 0} for class_name in selected_classes
    }

    for sample in samples:
        split_counts[sample.split] += 1
        class_counts[sample.class_name][sample.split] += 1

    summary = {
        "source_jpeg_dir": str(jpeg_dir),
        "output_root": str(output_root),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "resize_size": args.resize_size,
        "resize_method": args.resize_method if args.resize_size is not None else None,
        "jpeg_quality": args.jpeg_quality if args.resize_size is not None else None,
        "max_images_per_class": args.max_images_per_class,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_classes": len(selected_classes),
        "num_images": len(samples),
        "split_counts": split_counts,
        "classes": selected_classes,
        "class_split_counts": class_counts,
    }

    path = output_root / "subset_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_readme(output_root: Path) -> Path:
    path = output_root / "README_subset.md"
    text = """# AwA2 Subset

This folder is a portable subset generated from the original Animals with
Attributes 2 image dataset.

Files:

- `JPEGImages/`: copied subset images organized by class.
- `awa2_manifest_subset.csv`: relative image paths, labels, class names and split.
- `class_to_idx_subset.csv`: stable class-to-label mapping.
- `subset_summary.json`: seed, ratios, counts and source path used to generate it.

The manifest uses paths relative to this folder, so the subset can be moved to a
different machine without rewriting the CSV.

If the subset was generated with `--resize-size`, images were resized before
being written. The resize configuration is stored in `subset_summary.json`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def create_zip(output_root: Path) -> Path:
    zip_path = output_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_root.parent))
    return zip_path


def print_plan(
    selected_classes: list[str],
    class_to_images: dict[str, list[Path]],
    max_images_per_class: int,
) -> None:
    print("Selected classes:")
    for class_name in selected_classes:
        available = len(class_to_images[class_name])
        selected = min(available, max_images_per_class)
        print(f"  {class_name}: {selected}/{available} images")


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    if args.max_images_per_class <= 0:
        raise ValueError("--max-images-per-class must be positive")
    if args.resize_size is not None and args.resize_size <= 0:
        raise ValueError("--resize-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.resize_size is not None and args.copy_mode != "copy":
        raise ValueError("--resize-size requires --copy-mode copy")

    jpeg_dir = find_jpeg_images_dir(args.source_root)
    class_to_images = collect_class_images(jpeg_dir)
    selected_classes = choose_classes(args, class_to_images)
    print(f"Using JPEGImages directory: {jpeg_dir}")
    print_plan(selected_classes, class_to_images, args.max_images_per_class)

    if args.dry_run:
        print("Dry run complete. No files copied.")
        return 0

    output_root = prepare_output_root(args.output_root, args.allow_existing)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}
    rng = random.Random(args.seed)
    samples: list[CopiedSample] = []

    for class_name in selected_classes:
        class_images = list(class_to_images[class_name])
        rng.shuffle(class_images)
        selected_images = sorted(class_images[: args.max_images_per_class])
        split_to_images = split_images(
            selected_images,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            rng=rng,
        )

        for split_name in ("train", "val", "test"):
            for source_path in split_to_images[split_name]:
                destination = output_root / "JPEGImages" / class_name / source_path.name
                copy_image(
                    source=source_path,
                    destination=destination,
                    copy_mode=args.copy_mode,
                    resize_size=args.resize_size,
                    resize_method=args.resize_method,
                    jpeg_quality=args.jpeg_quality,
                )
                relative_path = destination.relative_to(output_root).as_posix()
                samples.append(
                    CopiedSample(
                        filepath=relative_path,
                        label=class_to_idx[class_name],
                        class_name=class_name,
                        split=split_name,
                    )
                )

    manifest_path = write_manifest(samples, output_root)
    class_map_path = write_class_map(class_to_idx, output_root)
    summary_path = write_summary(args, output_root, jpeg_dir, selected_classes, samples)
    readme_path = write_readme(output_root)

    print(f"Wrote subset: {output_root}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote class map: {class_map_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote README: {readme_path}")
    print(f"Total images copied/linked: {len(samples)}")

    if args.make_zip:
        zip_path = create_zip(output_root)
        print(f"Wrote zip archive: {zip_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
