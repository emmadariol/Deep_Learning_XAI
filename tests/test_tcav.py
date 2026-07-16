"""Regression tests for validated TCAV utilities."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.tcav import (
    ConceptSampleSelection,
    adjust_p_values,
    paired_permutation_p_value,
    score_cav_from_gradients,
    split_concept_selection,
    train_cav,
)


class _SampleDataset:
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(label=label, class_name=f"class_{label}")
            for label in (0, 0, 1, 1, 2, 2, 3, 3)
        ]

    def __len__(self) -> int:
        return len(self.samples)


class TCAVValidationTests(unittest.TestCase):
    def test_class_disjoint_split_has_no_label_overlap(self) -> None:
        dataset = _SampleDataset()
        selection = ConceptSampleSelection(
            concept_name="example",
            concept_index=0,
            positive_indices=[0, 1, 2, 3],
            negative_indices=[4, 5, 6, 7],
            positive_threshold=0.75,
            negative_threshold=0.25,
            positive_classes="class_0; class_1",
            negative_classes="class_2; class_3",
        )
        split = split_concept_selection(
            dataset,
            selection,
            validation_fraction=0.5,
            seed=7,
            prefer_class_disjoint=True,
        )
        self.assertEqual(split.strategy, "class_disjoint")
        for train_indices, validation_indices in (
            (split.positive_train_indices, split.positive_validation_indices),
            (split.negative_train_indices, split.negative_validation_indices),
        ):
            train_labels = {dataset.samples[index].label for index in train_indices}
            validation_labels = {dataset.samples[index].label for index in validation_indices}
            self.assertTrue(train_labels.isdisjoint(validation_labels))

    def test_cav_reports_held_out_accuracy(self) -> None:
        positive_train = torch.tensor([[3.0, 0.1], [2.5, 0.2], [3.2, -0.1]])
        negative_train = torch.tensor([[-3.0, 0.0], [-2.5, -0.2], [-3.2, 0.1]])
        positive_validation = torch.tensor([[2.8, 0.0], [3.1, 0.2]])
        negative_validation = torch.tensor([[-2.8, 0.0], [-3.1, -0.2]])
        cav = train_cav(
            positive_train,
            negative_train,
            concept_name="separable",
            layer_name="layer",
            positive_classes="positive",
            negative_classes="negative",
            epochs=120,
            lr=0.05,
            seed=3,
            positive_validation_activations=positive_validation,
            negative_validation_activations=negative_validation,
        )
        self.assertGreaterEqual(cav.validation_accuracy, 0.99)
        self.assertEqual(cav.positive_validation_count, 2)
        self.assertEqual(cav.negative_validation_count, 2)

    def test_directional_score_and_corrections_are_bounded(self) -> None:
        gradients = torch.tensor([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
        score = score_cav_from_gradients(gradients, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(score.tcav_score, 2.0 / 3.0, places=6)
        p_value = paired_permutation_p_value(
            [1.0, 1.0, 0.9, 1.0],
            [0.0, 0.1, 0.0, 0.1],
            max_permutations=1000,
        )
        adjusted = adjust_p_values([p_value, 0.5], method="benjamini_hochberg")
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertLessEqual(adjusted[0], adjusted[1])


if __name__ == "__main__":
    unittest.main()
