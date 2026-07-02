"""Create a tiny synthetic AwA2-like dataset for local smoke tests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging

LOGGER = logging.getLogger("create_sample_awa2")

DEFAULT_CLASSES = ("antelope", "grizzly+bear", "zebra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small synthetic JPEGImages/ tree for smoke tests."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "AWA2",
        help="Output root where JPEGImages/ will be created.",
    )
    parser.add_argument("--images-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def make_image(class_index: int, image_index: int, size: int) -> Image.Image:
    """Build a deterministic RGB image with class-specific colors and shapes."""
    base_color = (
        (60 + class_index * 55) % 256,
        (100 + image_index * 23) % 256,
        (150 + class_index * 35 + image_index * 7) % 256,
    )
    image = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(image)

    margin = 24 + image_index * 3
    box = (margin, margin, size - margin, size - margin)
    outline = tuple(255 - channel for channel in base_color)
    draw.rectangle(box, outline=outline, width=6)
    draw.ellipse(
        (
            size // 4,
            size // 4,
            size // 4 + 36 + class_index * 12,
            size // 4 + 36 + image_index * 4,
        ),
        fill=outline,
    )
    draw.text((12, 12), f"c{class_index} i{image_index}", fill=outline)
    return image


def create_sample_dataset(
    output_root: Path,
    class_names: tuple[str, ...],
    images_per_class: int,
    image_size: int,
) -> Path:
    if images_per_class < 3:
        raise ValueError("--images-per-class must be at least 3")
    if image_size < 64:
        raise ValueError("--image-size must be at least 64")

    jpeg_dir = output_root.expanduser().resolve() / "JPEGImages"
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    for class_index, class_name in enumerate(class_names):
        class_dir = jpeg_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(images_per_class):
            image = make_image(class_index, image_index, image_size)
            image_path = class_dir / f"{class_name}_{image_index:03d}.jpg"
            image.save(image_path, quality=90)

    return jpeg_dir


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    jpeg_dir = create_sample_dataset(
        output_root=args.output_root,
        class_names=DEFAULT_CLASSES,
        images_per_class=args.images_per_class,
        image_size=args.image_size,
    )
    LOGGER.info("Wrote sample JPEGImages directory: %s", jpeg_dir)


if __name__ == "__main__":
    main()
