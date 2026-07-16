"""Regression tests for idempotent AwA2 subset preparation."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_awa2 import (
    split_images,
    validate_existing_subset_configuration,
)


class AwA2PreparationTests(unittest.TestCase):
    def test_zero_ratio_produces_no_samples_for_that_split(self) -> None:
        images = [Path(f"image_{index}.jpg") for index in range(10)]
        splits = split_images(
            images,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
            rng=random.Random(42),
            ensure_eval_splits=True,
        )
        self.assertEqual(len(splits["test"]), 0)
        self.assertEqual(sum(len(values) for values in splits.values()), len(images))

    def test_allow_existing_configuration_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stored = {
                "seed": 42,
                "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            }
            (root / "subset_summary.json").write_text(
                json.dumps(stored),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_existing_subset_configuration(
                    root,
                    {
                        "seed": 7,
                        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    },
                )


if __name__ == "__main__":
    unittest.main()
