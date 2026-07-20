"""Tests for contrastive misclassification diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from src.misclassification import (
    concept_evidence_rows,
    replace_top_salient_pixels,
    saliency_pair_diagnostics,
    save_cbm_error_decomposition_figure,
    score_target_pair,
)


class TargetPairScoreTests(unittest.TestCase):
    def test_margin_is_wrong_logit_minus_true_logit(self) -> None:
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        scores = score_target_pair(
            logits,
            true_targets=torch.tensor([0]),
            wrong_targets=torch.tensor([1]),
        )
        self.assertAlmostEqual(float(scores.true_logits[0]), 1.0)
        self.assertAlmostEqual(float(scores.wrong_logits[0]), 3.0)
        self.assertAlmostEqual(float(scores.margins[0]), 2.0)
        self.assertGreater(float(scores.wrong_probabilities[0]), float(scores.true_probabilities[0]))


class SaliencyInterventionTests(unittest.TestCase):
    def test_identical_maps_have_perfect_overlap_and_rank_correlation(self) -> None:
        maps = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        iou, correlation = saliency_pair_diagnostics(maps, maps, top_fraction=0.25)
        self.assertAlmostEqual(float(iou[0]), 1.0)
        self.assertAlmostEqual(float(correlation[0]), 1.0)

    def test_deletion_replaces_only_the_requested_top_fraction(self) -> None:
        inputs = torch.zeros((1, 1, 2, 2))
        baseline = torch.ones_like(inputs)
        maps = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        deleted = replace_top_salient_pixels(inputs, maps, fraction=0.25, baseline=baseline)
        self.assertEqual(int((deleted == 1).sum().item()), 1)
        self.assertEqual(float(deleted[0, 0, 1, 1]), 1.0)


class ConceptEvidenceTests(unittest.TestCase):
    def test_linear_head_contributions_and_corrections_have_explicit_signs(self) -> None:
        class_head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            class_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        rows = concept_evidence_rows(
            class_head=class_head,
            concept_probabilities=torch.tensor([[0.5, 0.25]]),
            true_prototypes=torch.tensor([[1.0, 0.0]]),
            wrong_prototypes=torch.tensor([[0.0, 1.0]]),
            true_targets=torch.tensor([0]),
            wrong_targets=torch.tensor([1]),
            concept_names=["true_feature", "wrong_feature"],
        )
        by_name = {str(row["concept"]): row for row in rows}
        self.assertAlmostEqual(
            float(by_name["true_feature"]["wrong_vs_true_margin_contribution"]),
            -0.5,
        )
        self.assertAlmostEqual(
            float(by_name["wrong_feature"]["wrong_vs_true_margin_contribution"]),
            0.5,
        )
        self.assertLess(float(by_name["true_feature"]["correction_margin_delta"]), 0.0)
        self.assertGreater(float(by_name["true_feature"]["correction_true_probability_delta"]), 0.0)

    def test_bias_and_concepts_reconstruct_the_exact_linear_margin(self) -> None:
        class_head = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            class_head.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 2.0]]))
            class_head.bias.copy_(torch.tensor([-0.2, 0.4]))
        rows = concept_evidence_rows(
            class_head=class_head,
            concept_probabilities=torch.tensor([[0.4, 0.8]]),
            true_prototypes=torch.tensor([[1.0, 0.0]]),
            wrong_prototypes=torch.tensor([[0.0, 1.0]]),
            true_targets=torch.tensor([0]),
            wrong_targets=torch.tensor([1]),
            concept_names=["first", "second"],
        )
        contribution_sum = sum(
            float(row["wrong_vs_true_margin_contribution"]) for row in rows
        )
        reconstructed = contribution_sum + float(rows[0]["wrong_vs_true_bias"])
        self.assertAlmostEqual(reconstructed, float(rows[0]["original_cbm_margin"]), places=6)
        self.assertAlmostEqual(float(rows[0]["margin_reconstruction_error"]), 0.0, places=6)

    def test_four_panel_error_figure_is_written(self) -> None:
        class_head = nn.Linear(2, 2, bias=True)
        rows = concept_evidence_rows(
            class_head=class_head,
            concept_probabilities=torch.tensor([[0.4, 0.8]]),
            true_prototypes=torch.tensor([[1.0, 0.0]]),
            wrong_prototypes=torch.tensor([[0.0, 1.0]]),
            true_targets=torch.tensor([0]),
            wrong_targets=torch.tensor([1]),
            concept_names=["first", "second"],
        )
        summary = {
            "true_class": "true",
            "cbm_predicted_class": "wrong",
            "cbm_confidence": 0.7,
            "cbm_true_probability": 0.3,
            "cbm_predicted_probability": 0.7,
            "wrong_vs_true_margin": rows[0]["original_cbm_margin"],
            "wrong_vs_true_bias": rows[0]["wrong_vs_true_bias"],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "decomposition.png"
            save_cbm_error_decomposition_figure(
                image=torch.zeros(3, 8, 8),
                rows=rows,
                summary=summary,
                output_path=output,
                top_k=2,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
