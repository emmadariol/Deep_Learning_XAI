"""Regression tests for manifest mapping and preprocessing policy."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torchvision import transforms

from src.data import build_resnet_transforms, load_class_names


class DataMappingTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(path)

    def test_manifest_rejects_duplicate_class_with_different_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            self._write_image(image_a)
            self._write_image(image_b)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "class_name", "split"])
                writer.writeheader()
                writer.writerow({"filepath": str(image_a), "label": 0, "class_name": "antelope", "split": "train"})
                writer.writerow({"filepath": str(image_b), "label": 1, "class_name": "antelope", "split": "val"})

            with self.assertRaises(ValueError):
                load_class_names(manifest)

    def test_train_and_eval_transforms_are_intentionally_different(self) -> None:
        train_transform = build_resnet_transforms(train=True)
        eval_transform = build_resnet_transforms(train=False)
        self.assertIsInstance(train_transform.transforms[0], transforms.RandomResizedCrop)
        self.assertIsInstance(train_transform.transforms[1], transforms.RandomHorizontalFlip)
        self.assertIsInstance(eval_transform.transforms[0], transforms.Resize)
        self.assertIsInstance(eval_transform.transforms[1], transforms.CenterCrop)


if __name__ == "__main__":
    unittest.main()
