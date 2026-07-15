"""Regression tests for concept transition summaries."""

from __future__ import annotations

import unittest

import numpy as np

from src.concepts import AwA2ConceptBank, concept_transition_summary


class ConceptTransitionTests(unittest.TestCase):
    def test_gained_and_lost_concepts_respect_delta_sign(self) -> None:
        bank = AwA2ConceptBank(
            class_names=["source", "target"],
            concept_names=["fur", "horns", "water", "tail"],
            matrix=np.array(
                [
                    [0.8, 0.1, 0.7, 0.4],
                    [0.2, 0.9, 0.6, 0.4],
                ],
                dtype=float,
            ),
        )
        summary = concept_transition_summary(bank, "source", "target", top_k=4)
        gained = str(summary["gained_concepts"])
        lost = str(summary["lost_concepts"])

        self.assertIn("horns", gained)
        self.assertNotIn("fur", gained)
        self.assertIn("fur", lost)
        self.assertNotIn("horns", lost)


if __name__ == "__main__":
    unittest.main()
